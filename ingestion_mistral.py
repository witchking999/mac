#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline vLLM (Python API) – Mistral Small 3.2 24B (multimodal + structured JSON + one tool)
- Two-user-prompt format per file: one for TXT, one for ATTACHMENT
- Converts ALL attachments to PNG (images/PDF; Office→PDF→PNG when possible)
- Only tool exposed: calculate_provenance_sha
- Validates structured JSON with Pydantic
- Writes per-file JSONL and combined JSONL
- NEW: ThreadPool batching with IN_FLIGHT parallel batches to keep GPUs saturated

Run (inside vLLM container):
  apt-get update && apt-get install -y poppler-utils libreoffice-common libreoffice-writer libreoffice-calc
  pip install vllm pillow pdf2image python-dotenv pydantic

  python mistral_offline_ingestor.py /path/to/*_mapped_text --case-id CASE123
"""

from __future__ import annotations
import os, io, json, base64, hashlib, argparse, logging, subprocess, shutil, threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from PIL import Image
from pdf2image import convert_from_path

from vllm import LLM
from vllm.sampling_params import SamplingParams

# ──────────────────────────────────────────────────────────────────────────────
# Env / logging
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(os.environ.get("ENV_FILE", ".env"))
logger = logging.getLogger("mistral_offline_ingestion")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MODEL_ID   = os.getenv("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

# Batch/concurrency knobs
BATCH = int(os.getenv("BATCH", "96"))          # files per batch (64–128 good)
IN_FLIGHT = int(os.getenv("IN_FLIGHT", "3"))   # concurrent batches (2–4 good)

WRITE_LOCK = threading.Lock()   # to serialize JSONL writes across threads

# ──────────────────────────────────────────────────────────────────────────────
# Schema (structured JSON target)
# ──────────────────────────────────────────────────────────────────────────────
class EmailNode(BaseModel):
    type: Literal[
        "SENDER","RECIPIENT","DATE","MESSAGE","ATTACHMENT","LINK",
        "SUBJECT","MENTIONS_PERSON","CC","MENTIONS_ORGANIZATION"
    ]
    value: str
    confidence: float = Field(ge=0, le=1, default=0.9)

class EmailRelationship(BaseModel):
    type: Literal[
        "HAS_SENDER","HAS_RECIPIENT","HAS_SUBJECT","SENT_ON_DATE",
        "HAS_ATTACHMENT","HAS_LINK","MENTIONS_PERSON","CC","MENTIONS_ORGANIZATION"
    ]
    source: str
    target: str

class EmailEntityExtraction(BaseModel):
    nodes: List[EmailNode]
    relationships: List[EmailRelationship]

# ──────────────────────────────────────────────────────────────────────────────
# Only tool exposed to the model: calculate_provenance_sha
# ──────────────────────────────────────────────────────────────────────────────
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def calculate_provenance_sha(nodes: List[Dict[str, Any]],
                             relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    subject = next((n.get("value") for n in nodes if n.get("type")=="SUBJECT" and n.get("value")), "No Subject")
    parent_content = f"EMAIL_SUBJECT:{subject}"
    parent_sha = _sha256(parent_content)

    node_shas = []
    for n in nodes:
        if not n.get("value"): continue
        base = f"{n.get('type','')}:{n.get('value','')}"
        s = _sha256(base)
        node_shas.append({
            "entity": n, "sha256": s, "parent_sha": parent_sha,
            "content_hash": _sha256(base),
            "relationship_hash": _sha256(f"{s}:{parent_sha}"),
            "timestamp": datetime.now().isoformat()
        })
    rel_shas = []
    for r in relationships:
        if not (r.get("source") and r.get("target")): continue
        base = f"{r.get('type','')}:{r.get('source','')}->{r.get('target','')}"
        s = _sha256(base)
        rel_shas.append({
            "relationship": r, "sha256": s, "parent_sha": parent_sha,
            "content_hash": _sha256(base),
            "relationship_hash": _sha256(f"{s}:{parent_sha}"),
            "timestamp": datetime.now().isoformat()
        })

    return {
        "parent_sha": parent_sha,
        "parent_content": parent_content,
        "node_shas": node_shas,
        "relationship_shas": rel_shas,
    }

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculate_provenance_sha",
        "description": "Compute SHA-256 provenance hierarchy for nodes and relationships.",
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "object"}},
                "relationships": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["nodes", "relationships"]
        }
    }
}]

# ──────────────────────────────────────────────────────────────────────────────
# Utility: write JSONL
# ──────────────────────────────────────────────────────────────────────────────
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# Attachment → PNG conversion
# ──────────────────────────────────────────────────────────────────────────────
def _downscale_to_png_bytes(img: Image.Image, max_side: int = 1024) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def image_file_to_png_data_uri(path: str) -> Optional[str]:
    try:
        with Image.open(path) as im:
            b = _downscale_to_png_bytes(im, 1024)
            return f"data:image/png;base64,{base64.b64encode(b).decode('utf-8')}"
    except Exception as e:
        logger.error(f"image to png failed for {path}: {e}")
        return None

def pdf_to_png_data_uris(path: str, max_pages: int = 2) -> List[str]:
    uris = []
    try:
        pages = convert_from_path(path, dpi=165, thread_count=2, first_page=1, last_page=max_pages)
        for im in pages[:max_pages]:
            b = _downscale_to_png_bytes(im, 1024)
            uris.append(f"data:image/png;base64,{base64.b64encode(b).decode('utf-8')}")
    except Exception as e:
        logger.error(f"pdf->png failed for {path}: {e}")
    return uris

def office_to_pdf(path: str) -> Optional[str]:
    """Use LibreOffice headless to convert to PDF next to the file."""
    if not shutil.which("libreoffice"):
        return None
    try:
        outdir = str(Path(path).parent)
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", outdir, path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        pdf_path = str(Path(outdir) / (Path(path).stem + ".pdf"))
        return pdf_path if Path(pdf_path).exists() else None
    except Exception as e:
        logger.error(f"libreoffice convert failed for {path}: {e}")
        return None

def any_attachment_to_png_data_uris(path: str) -> List[str]:
    ext = Path(path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}:
        uri = image_file_to_png_data_uri(path)
        return [uri] if uri else []
    if ext == ".pdf":
        return pdf_to_png_data_uris(path, max_pages=2)
    if ext in {".doc", ".docx", ".xls", ".xlsx"}:
        pdf = office_to_pdf(path)
        if pdf:
            return pdf_to_png_data_uris(pdf, max_pages=2)
    # fallback: base64 as generic (keeps “convert all” spirit)
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "application/octet-stream"
        return [f"data:{mime};base64,{b64}"]
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# vLLM offline init (GPU fast path)
# ──────────────────────────────────────────────────────────────────────────────
_LLM = None
_SAMPLING = SamplingParams(max_tokens=900, temperature=0.0, top_p=0.95, n=1)

def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=2,        # use both GPUs (set to 1 if single GPU)
            max_model_len=6144,
            gpu_memory_utilization=0.94,
        )
    return _LLM

# ──────────────────────────────────────────────────────────────────────────────
# Two-user-prompt format per file + ONLY one tool (calculate_provenance_sha)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an email analysis expert. You will receive two user messages:\n"
    "  1) a TEXT extraction request (plain email text)\n"
    "  2) an ATTACHMENT extraction request (converted to PNG data URLs)\n"
    "Extract ENTITIES and RELATIONSHIPS in JSON ONLY using this schema keys:\n"
    "  nodes: [{type, value, confidence?}],\n"
    "  relationships: [{type, source, target}].\n"
    "Entity types: SENDER, RECIPIENT, DATE, MESSAGE, ATTACHMENT, LINK, SUBJECT, MENTIONS_PERSON, CC, MENTIONS_ORGANIZATION.\n"
    "Relationship types: HAS_SENDER, HAS_RECIPIENT, HAS_SUBJECT, SENT_ON_DATE, HAS_ATTACHMENT, HAS_LINK, MENTIONS_PERSON, CC, MENTIONS_ORGANIZATION.\n"
    "After you provide nodes and relationships, CALL the function calculate_provenance_sha with those arrays.\n"
    "Do not call any other tool."
)

def build_messages_for_text(text: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "TEXT EXTRACTION: Extract nodes and relationships from this email body. Return JSON only."},
            {"type": "text", "text": text},
        ],
    }

def build_messages_for_attachment(path: str) -> Dict[str, Any]:
    uris = any_attachment_to_png_data_uris(path)
    content = [{"type":"text","text":"ATTACHMENT EXTRACTION: Describe the attachment and extract nodes/relationships. Return JSON only."}]
    for u in uris:
        content.append({"type":"image_url","image_url":{"url": u}})
    if len(content) == 1:
        content.append({"type":"text","text":"(Attachment conversion failed; describe based on filename.)"})
    return {"role": "user", "content": content}

def build_conversation_for_file(file_path: str) -> List[Dict[str, Any]]:
    """Return the full conversation (system + 2 user turns) for a single file."""
    ext = Path(file_path).suffix.lower()
    msgs = [{"role":"system", "content": SYSTEM_PROMPT}]
    if ext in {".txt", ".html"}:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        msgs.append(build_messages_for_text(text))
        msgs.append(build_messages_for_attachment(file_path))
    else:
        msgs.append(build_messages_for_text("(No email body provided; focus on attachment context if any.)"))
        msgs.append(build_messages_for_attachment(file_path))
    return msgs

def parse_choice_to_record(choice) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
    """Parse a single vLLM choice (with optional tool_calls) into nodes, rels, sha_result, raw_text."""
    tool_calls = getattr(choice, "tool_calls", None)
    raw_text = choice.text or ""

    nodes, rels = [], []
    if raw_text.strip():
        try:
            parsed = json.loads(raw_text)
            nodes = parsed.get("nodes", [])
            rels  = parsed.get("relationships", [])
        except Exception:
            pass

    sha_result = None
    if tool_calls:
        for tc in tool_calls:
            if tc.function.name == "calculate_provenance_sha":
                args = json.loads(tc.function.arguments or "{}")
                n = args.get("nodes", nodes)
                r = args.get("relationships", rels)
                sha_result = calculate_provenance_sha(n, r)
                nodes, rels = n, r

    if sha_result is None:
        sha_result = calculate_provenance_sha(nodes, rels)

    # Validate with Pydantic
    validated = EmailEntityExtraction.model_validate({
        "nodes": [n for n in nodes if n.get("value")],
        "relationships": [r for r in rels if r.get("source") and r.get("target")],
    }).model_dump()

    return validated["nodes"], validated["relationships"], sha_result, raw_text

# ──────────────────────────────────────────────────────────────────────────────
# Batch + JSONL with ThreadPool
# ──────────────────────────────────────────────────────────────────────────────
def append_per_file_and_combined(case_id: str, bundle_dir: str, fp: str, combined: str,
                                 nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]], sha: Dict[str, Any]) -> None:
    rec = {
        "case_id": case_id,
        "email_bundle": Path(bundle_dir).name,
        "source_directory": bundle_dir,
        "file_path": fp,
        "extracted_at": datetime.now().isoformat(),
        "nodes": nodes,
        "relationships": rels,
        "sha_provenance": sha,
    }
    per_file = os.path.join(OUTPUT_DIR, case_id, "files", Path(bundle_dir).name, Path(fp).name + ".jsonl")
    with WRITE_LOCK:
        append_jsonl(per_file, rec)
        append_jsonl(combined, rec)

def run_batch(llm: LLM,
              batch_items: List[Tuple[str, str]],   # list of (bundle_dir, file_path)
              combined: str,
              case_id: str) -> Dict[str, int]:
    """Process one batch: build conversations, call llm.chat once, write JSONL for every file."""
    convs = [build_conversation_for_file(fp) for (_, fp) in batch_items]
    outs = llm.chat(messages=convs, sampling_params=_SAMPLING, tools=TOOLS)

    processed = errors = skipped = 0
    for (bundle_dir, fp), req_out in zip(batch_items, outs):
        try:
            choice = req_out.outputs[0]
            nodes, rels, sha, raw = parse_choice_to_record(choice)
            append_per_file_and_combined(case_id, bundle_dir, fp, combined, nodes, rels, sha)
            processed += 1
        except Exception as e:
            logger.error(f"[BATCH] Failed {fp}: {e}")
            errors += 1

    return {"processed_count": processed, "error_count": errors, "skipped_count": skipped}

def process_all_files_threadpooled(case_id: str,
                                   all_files: List[Tuple[str, str]],
                                   combined: str) -> Dict[str, int]:
    """Submit batches with ThreadPool to keep 2×A100 saturated."""
    llm = get_llm()
    totals = {"processed":0,"errors":0,"skipped":0}

    with ThreadPoolExecutor(max_workers=IN_FLIGHT) as ex:
        futures = []
        for i in range(0, len(all_files), BATCH):
            futures.append(ex.submit(run_batch, llm, all_files[i:i+BATCH], combined, case_id))
        for fut in as_completed(futures):
            stats = fut.result()
            totals["processed"] += stats["processed_count"]
            totals["errors"]    += stats["error_count"]
            totals["skipped"]   += stats["skipped_count"]

    return totals

# ──────────────────────────────────────────────────────────────────────────────
# Driver: gather files and run threadpool batches
# ──────────────────────────────────────────────────────────────────────────────
def run(case_id: str, mapped_dir: str, max_emails: Optional[int], batch_size: int) -> Dict[str, Any]:
    # Keep batch_size param for API compatibility; we use env-driven BATCH/IN_FLIGHT now
    os.makedirs(os.path.join(OUTPUT_DIR, case_id), exist_ok=True)
    combined = os.path.join(OUTPUT_DIR, case_id, f"{case_id}_all_extractions.jsonl")

    base = Path(mapped_dir)
    if not base.exists(): return {"status":"error","message":f"dir not found: {base}"}

    bundles = sorted([str(p) for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if max_emails is not None: bundles = bundles[:max(0, max_emails)]

    # flatten to [(bundle_dir, file_path), ...]
    all_files: List[Tuple[str, str]] = []
    for b in bundles:
        for fp in Path(b).glob("*"):
            if fp.name.startswith("._") or not fp.is_file(): continue
            all_files.append((b, str(fp)))

    if not all_files: return {"status":"completed","message":"Nothing to process"}

    totals = process_all_files_threadpooled(case_id, all_files, combined)

    summary = {
        "status": "success",
        "case_id": case_id,
        "batches": {"BATCH": BATCH, "IN_FLIGHT": IN_FLIGHT},
        "dirs_total": len(bundles),
        "files_total": len(all_files),
        "totals": totals,
        "outputs": {"combined_jsonl": combined, "per_file_root": os.path.join(OUTPUT_DIR, case_id, "files")},
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(OUTPUT_DIR, case_id, "overall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Offline vLLM Mistral email ingestor (two-user-prompt, one tool, batched)")
    ap.add_argument("directory", help="Directory containing *_mapped_text subdirs")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--max-emails", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=25)  # kept for compatibility; use env BATCH/IN_FLIGHT instead
    ap.add_argument("--rebuild-combined", action="store_true")
    args = ap.parse_args()

    if args.rebuild_combined:
        # Rebuild combined from per-file JSONLs
        root = os.path.join(OUTPUT_DIR, args.case_id, "files")
        combined = os.path.join(OUTPUT_DIR, args.case_id, f"{args.case_id}_all_extractions.jsonl")
        if os.path.exists(combined): os.remove(combined)
        for p in Path(root).rglob("*.jsonl"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    append_jsonl(combined, json.loads(line))
        print(json.dumps({"rebuilt_combined_jsonl": combined}, indent=2))
        return

    result = run(args.case_id, args.directory, args.max_emails, args.batch_size)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
