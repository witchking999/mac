#!/usr/bin/env python3
"""
Multimodal Email Ingestor (vLLM OpenAI-compatible client) - Optimized Batch Processing
- One request per email bundle (directory)
- Parses .txt and .msg (Outlook) email files + attachments
- Multimodal prompt (text + up to 10 images) sent to LLM
- Optional tool-call for SHA provenance (calculate_sha256)
- Pydantic-validated JSON output (nodes, relationships, attachments)
- Optimized batch processing with controlled concurrency for maximum GPU utilization

Optimized for dual GPU A100 setup:
- max_num_seqs: 64 (sequences per batch on vLLM server)
- max_num_batched_tokens: 8192 (total tokens per batch)
- tensor_parallel_size: 2 (across both GPUs)
- MAX_CONCURRENT_REQUESTS: 8 (concurrent API calls)
- BATCH_SIZE: 4 (emails per concurrent batch)

Env:
  OPENAI_BASE_URL=http://localhost:8000/v1
  OPENAI_MODEL=mistral-small-3.2-24b-instruct-2506
  OPENAI_API_KEY=docker
  OUTPUT_DIR=./output
  MAX_CONCURRENT_REQUESTS=8
  BATCH_SIZE=4
  MAX_INLINE_ATTACHMENT=2097152
"""

from __future__ import annotations
import os, sys, json, base64, hashlib, argparse, asyncio, mimetypes, logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
import requests
from openai import OpenAI, AsyncOpenAI

# For images & .msg:
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import extract_msg  # parse .msg (Outlook)
except Exception:
    extract_msg = None

# ──────────────────────────────────────────────────────────────────────────────
# RAPIDS-Optimized Config for Container Environment
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(os.getenv("ENVFILE", "/workspace/.env"))

logger = logging.getLogger("email_ingestor_rapids")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# Container-to-container networking for Mistral vLLM
# Fallback to OpenAI API if vLLM fails
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Use smaller, reliable model
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "/workspace/output")

# RAPIDS-optimized batch processing (higher concurrency for GPU efficiency)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "12"))  # Increased for RAPIDS
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "6"))  # Larger batches for GPU efficiency
MAX_INLINE_ATTACHMENT   = int(os.getenv("MAX_INLINE_ATTACHMENT", "2097152"))

# RAPIDS GPU memory management
try:
    import cudf
    import cuml
    import cugraph
    RAPIDS_AVAILABLE = True
    logger.info("✅ RAPIDS libraries loaded for GPU acceleration")
except ImportError as e:
    RAPIDS_AVAILABLE = False
    logger.warning(f"⚠️ RAPIDS not available: {e}")

# GPU memory manager for RAPIDS operations
class RAPIDSMemoryManager:
    def __init__(self):
        self.active_dataframes = []

    def track_dataframe(self, df):
        """Track cuDF DataFrames for memory management"""
        if RAPIDS_AVAILABLE and hasattr(df, 'memory_usage'):
            self.active_dataframes.append(df)

    def cleanup(self):
        """Clean up GPU memory"""
        for df in self.active_dataframes:
            try:
                del df
            except:
                pass
        self.active_dataframes.clear()
        if RAPIDS_AVAILABLE:
            try:
                import cudf
                cudf._memory_manager._default_memory_manager.free_all_blocks()
            except:
                pass

rapids_memory = RAPIDSMemoryManager()  # 2 MiB

# Clients for OpenAI-compatible vLLM/NIM server
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Fast HTTP session for direct API calls
http_session = requests.Session()
http_session.headers.update({
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
})
# Connection pooling for maximum performance
adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,  # Keep 20 connections open
    pool_maxsize=20,      # Max 20 connections per pool
    max_retries=3,
    pool_block=False
)
http_session.mount("http://", adapter)
GEN_PARAMS = dict(temperature=0.1, top_p=0.9, max_tokens=2048, stream=False)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schema
# ──────────────────────────────────────────────────────────────────────────────
class EmailNode(BaseModel):
    type: Literal["SENDER","RECIPIENT","DATE","MESSAGE","ATTACHMENT","LINK",
                  "SUBJECT","MENTIONS_PERSON","CC","MENTIONS_ORGANIZATION"]
    value: str
    confidence: float = Field(ge=0, le=1, default=0.9)
    mime: Optional[str] = None
    sha256: Optional[str] = None
    content_b64: Optional[str] = None

class EmailRelationship(BaseModel):
    type: Literal["HAS_SENDER","HAS_RECIPIENT","HAS_SUBJECT","SENT_ON_DATE",
                  "HAS_ATTACHMENT","HAS_LINK","MENTIONS_PERSON","CC","MENTIONS_ORGANIZATION"]
    source: str
    target: str

class Attachment(BaseModel):
    filename: str
    mime: str
    size_bytes: int
    sha256: str
    content_b64: Optional[str] = None

class EmailEntityExtraction(BaseModel):
    nodes: List[EmailNode]
    relationships: List[EmailRelationship]
    attachments: List[Attachment] = Field(default_factory=list)

    @classmethod
    def get_json_schema(cls):
        """Get the JSON schema for use with OpenAI's json_schema response format."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "email-entity-extraction",
                "schema": cls.model_json_schema(),
                "strict": True
            }
        }

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def file_to_data_url(path: str, mime: str) -> str:
    return f"data:{mime};base64,{file_to_b64(path)}"

def validate_image(path: str) -> bool:
    if Image is None: return True
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def calculate_sha256_local(content: str) -> Dict[str, Any]:
    return {"sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "timestamp": datetime.now().isoformat()}

TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"calculate_sha256",
            "description":"Calculate SHA-256 for provenance.",
            "parameters":{
                "type":"object",
                "properties":{"content":{"type":"string"}},
                "required":["content"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "You are an email analysis assistant. Extract ENTITIES and RELATIONSHIPS from email content. "
    "Return ONLY a valid JSON object with this EXACT structure:\n"
    "{\n"
    '  "nodes": [\n'
    '    {"type": "SENDER", "value": "sender name", "confidence": 0.9},\n'
    '    {"type": "RECIPIENT", "value": "recipient@example.com"},\n'
    '    {"type": "SUBJECT", "value": "email subject"},\n'
    '    {"type": "DATE", "value": "date string"},\n'
    '    {"type": "MESSAGE", "value": "message content"}\n'
    '  ],\n'
    '  "relationships": [\n'
    '    {"type": "HAS_SENDER", "source": "sender name", "target": "message"},\n'
    '    {"type": "HAS_RECIPIENT", "source": "recipient@example.com", "target": "message"},\n'
    '    {"type": "HAS_SUBJECT", "source": "email subject", "target": "message"},\n'
    '    {"type": "SENT_ON_DATE", "source": "date string", "target": "message"}\n'
    '  ],\n'
    '  "attachments": []\n'
    "}\n\n"
    "Use EXACTLY these type values: SENDER, RECIPIENT, DATE, MESSAGE, ATTACHMENT, LINK, SUBJECT, MENTIONS_PERSON, CC, MENTIONS_ORGANIZATION for nodes. "
    "Use EXACTLY these relationship types: HAS_SENDER, HAS_RECIPIENT, HAS_SUBJECT, SENT_ON_DATE, HAS_ATTACHMENT, HAS_LINK, MENTIONS_PERSON, CC, MENTIONS_ORGANIZATION. "
    "Each node must have 'type' and 'value' fields. Each relationship must have 'type', 'source', and 'target' fields. "
    "If helpful for provenance, call calculate_sha256 on important strings. "
    "Return ONLY the JSON object, no markdown formatting or explanations."
)

# ──────────────────────────────────────────────────────────────────────────────
# Collect text/images/attachments from a bundle (supports .txt and .msg)
# ──────────────────────────────────────────────────────────────────────────────
def parse_msg_file(path: Path) -> Dict[str, Any]:
    """Extract text body and attachments from .msg (Outlook)."""
    if extract_msg is None:
        logger.warning("extract-msg not installed; skipping .msg parsing")
        return {"text": "", "attachments": []}
    try:
        msg = extract_msg.Message(str(path))
        msg_sender = msg.sender or ""
        msg_subj   = msg.subject or ""
        msg_date   = str(msg.date) if msg.date else ""
        body = msg.body or msg.bodyHTML or ""
        text = f"MSG FILE: {path.name}\nSENDER: {msg_sender}\nSUBJECT: {msg_subj}\nDATE: {msg_date}\n\n{body}"
        atts: List[Attachment] = []
        for a in msg.attachments:
            try:
                fname = a.longFilename or a.shortFilename or f"attachment_{len(atts)}"
                raw = a.data
                mime = get_mime_type(fname)
                sha  = sha256_bytes(raw)
                size = len(raw)
                b64  = base64.b64encode(raw).decode("utf-8") if size <= MAX_INLINE_ATTACHMENT else None
                atts.append(Attachment(filename=fname, mime=mime, size_bytes=size, sha256=sha, content_b64=b64))
            except Exception as e:
                logger.warning(f"msg attachment error: {e}")
        return {"text": text, "attachments": atts}
    except Exception as e:
        logger.warning(f"parse_msg error {path}: {e}")
        return {"text": "", "attachments": []}

def collect_modal_inputs_from_dir(dir_path: Path) -> Dict[str, Any]:
    """RAPIDS-optimized file collection with GPU acceleration"""
    texts: List[str] = []
    image_paths: List[str] = []
    attachments: List[Attachment] = []

    # Use RAPIDS for parallel file processing if available
    if RAPIDS_AVAILABLE:
        try:
            # Create cuDF DataFrame for file metadata analysis
            file_info = []
            for p in dir_path.glob("*"):
                if p.name.startswith("._") or not p.is_file():
                    continue
                file_info.append({
                    'path': str(p),
                    'name': p.name,
                    'ext': p.suffix.lower(),
                    'mime': get_mime_type(str(p)),
                    'size': p.stat().st_size,
                    'is_image': get_mime_type(str(p)).startswith("image/")
                })

            if file_info:
                # GPU-accelerated DataFrame operations
                df = cudf.DataFrame(file_info)
                rapids_memory.track_dataframe(df)

                # RAPIDS-powered filtering
                text_files = df[df['ext'] == '.txt']
                msg_files = df[df['ext'] == '.msg']
                image_files = df[df['is_image'] == True]

                logger.info(f"📊 RAPIDS analyzed {len(df)} files: {len(text_files)} text, {len(msg_files)} msg, {len(image_files)} images")

        except Exception as e:
            logger.warning(f"RAPIDS file analysis failed: {e}")

    # Process files (fallback to CPU processing)
    for p in dir_path.glob("*"):
        if p.name.startswith("._") or not p.is_file():
            continue
        ext  = p.suffix.lower()
        mime = get_mime_type(str(p))
        size = p.stat().st_size

        if ext == ".txt":
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                texts.append(f"FILE: {p.name}\n{txt}")
            except Exception as e:
                logger.warning(f"Read text failed for {p}: {e}")

        elif ext == ".msg":
            parsed = parse_msg_file(p)
            if parsed["text"]:
                texts.append(parsed["text"])
            attachments.extend(parsed["attachments"])

        # images to send in prompt
        if mime.startswith("image/") and validate_image(str(p)):
            image_paths.append(str(p))

        # always append every file as a structured attachment too
        try:
            raw = p.read_bytes()
            sha = sha256_bytes(raw)
            b64 = base64.b64encode(raw).decode("utf-8") if size <= MAX_INLINE_ATTACHMENT else None
            attachments.append(Attachment(filename=p.name, mime=mime, size_bytes=size, sha256=sha, content_b64=b64))
        except Exception as e:
            logger.warning(f"Attachment read failed for {p}: {e}")

    return {"texts":texts, "image_paths":image_paths, "attachments":attachments}

def build_bundle_messages(texts: List[str], image_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if texts:
        content.append({"type":"text",
                        "text": "Extract structured entities/relationships and return ONLY strict JSON.\n\nEMAIL TEXT:\n" +
                                "\n\n".join(texts)})
    added = 0
    for p in image_paths:
        if added >= 10: break
        mime = get_mime_type(p)
        if not mime.startswith("image/"): continue
        try:
            content.append({"type":"image_url", "image_url":{"url": file_to_data_url(p, mime)}})
            added += 1
        except Exception as e:
            logger.warning(f"Skip image {p}: {e}")
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":content}]

async def process_emails_concurrent_async(email_dirs: List[str]) -> List[Dict[str, Any]]:
    """Process emails with optimized batch processing for maximum GPU utilization."""
    logger.info(f"⚡ Processing {len(email_dirs)} emails with optimized batched concurrency")
    logger.info(f"   Batch size: {BATCH_SIZE}, Max concurrent: {MAX_CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_email_batch(email_batch: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of emails concurrently."""
        async with semaphore:
            # Create tasks for this batch
            tasks = []
            for email_dir in email_batch:
                task = process_single_email_async(email_dir)
                tasks.append(task)

            # Execute batch concurrently
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                # Handle any exceptions in results
                processed_results = []
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        # Create error result for failed email
                        dpath = Path(email_batch[i])
                        processed_results.append({
                            "error": f"Processing failed: {str(result)}",
                            "email_bundle": dpath.name,
                            "source_directory": email_batch[i]
                        })
                    else:
                        processed_results.append(result)
                return processed_results
            except Exception as e:
                # Handle batch-level failures
                error_results = []
                for email_dir in email_batch:
                    dpath = Path(email_dir)
                    error_results.append({
                        "error": f"Batch processing failed: {str(e)}",
                        "email_bundle": dpath.name,
                        "source_directory": email_dir
                    })
                return error_results

    async def process_single_email_async(email_dir: str) -> Dict[str, Any]:
        """Process a single email using AsyncOpenAI directly."""
        dpath = Path(email_dir)
        modal = collect_modal_inputs_from_dir(dpath)
        messages = build_bundle_messages(modal["texts"], modal["image_paths"])
        result = await extract_bundle_async(messages)

        # Merge attachments
        if "error" not in result:
            result.setdefault("attachments", [])
            have = {(a.get("filename"), a.get("sha256")) for a in result["attachments"]}
            for a in modal["attachments"]:
                key = (a.filename, a.sha256)
                if key not in have:
                    result["attachments"].append(a.model_dump())
                    have.add(key)

        # Annotate with directory
        result["email_bundle"] = dpath.name
        result["source_directory"] = email_dir
        return result

    # Split email directories into optimized batches
    batches = [email_dirs[i:i + BATCH_SIZE] for i in range(0, len(email_dirs), BATCH_SIZE)]
    logger.info(f"   Created {len(batches)} batches for processing")

    # Process all batches concurrently (but limited by semaphore)
    batch_tasks = [process_email_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    # Flatten results from all batches
    all_results = []
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            all_results.extend(batch_result)
        else:
            # Handle batch-level exception
            logger.error(f"Batch processing failed: {batch_result}")

    successful = sum(1 for r in all_results if "error" not in r)
    failed = len(all_results) - successful
    logger.info(f"   ✓ Processed {successful} emails successfully, {failed} failed")

    return all_results

async def extract_bundle_async(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Async version of extract_bundle using AsyncOpenAI for true concurrency."""
    # First call: allow optional tool calls
    try:
        # Define the JSON schema for guided output
        json_schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["type", "value"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "source": {"type": "string"},
                            "target": {"type": "string"}
                        },
                        "required": ["type", "source", "target"]
                    }
                },
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "mime": {"type": "string"},
                            "size_bytes": {"type": "integer"},
                            "sha256": {"type": "string"},
                            "content_b64": {"type": "string"}
                        },
                        "required": ["filename", "mime", "size_bytes", "sha256"]
                    }
                }
            },
            "required": ["nodes", "relationships", "attachments"]
        }

        first = await async_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            response_format={"type":"json_object"},
            **GEN_PARAMS
        )
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    provenance = []
    tool_calls = first.choices[0].message.tool_calls or []

    # Try direct JSON parse first
    if not tool_calls and (first.choices[0].message.content or "").strip():
        text = (first.choices[0].message.content or "").strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        try:
            payload = json.loads(text)
            obj = EmailEntityExtraction.model_validate(payload).model_dump()
            obj["provenance"] = provenance
            return obj
        except Exception:
            # Fall through to second attempt
            pass

    # If tools exist or direct parse failed, do second call
    follow = messages.copy()
    if tool_calls:
        follow.append({"role": "assistant", "tool_calls": tool_calls})
        for tc in tool_calls:
            if tc.type == "function" and tc.function.name == "calculate_sha256":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    out = calculate_sha256_local(args.get("content", ""))
                    provenance.append({"tool_call_id": tc.id, **out})
                    follow.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": json.dumps(out)})
                except Exception as e:
                    follow.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": json.dumps({"error": str(e)})})

    # Second call for final JSON
    try:
        final = await async_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=follow + [{"role": "user", "content": "Use any tool results above and return ONLY the JSON object."}],
            response_format={"type":"json_object"},
            **GEN_PARAMS
        )
    except Exception as e:
        return {"error": f"API follow-up failed: {e}"}

    text = (final.choices[0].message.content or "").strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"): text = text[:-3]
    try:
        payload = json.loads(text)
        obj = EmailEntityExtraction.model_validate(payload).model_dump()
        obj["provenance"] = provenance
        return obj
    except Exception as e:
        return {"error": f"validate/parse failed: {e}", "raw_response": text[:500], "provenance": provenance}

async def process_email_directory_once_async(email_dir: str) -> Dict[str, Any]:
    """Async version of process_email_directory_once."""
    dpath = Path(email_dir)
    modal = collect_modal_inputs_from_dir(dpath)
    messages = build_bundle_messages(modal["texts"], modal["image_paths"])
    result = await extract_bundle_async(messages)

    # Merge attachments
    if "error" not in result:
        result.setdefault("attachments", [])
        have = {(a.get("filename"), a.get("sha256")) for a in result["attachments"]}
        for a in modal["attachments"]:
            key = (a.filename, a.sha256)
            if key not in have:
                result["attachments"].append(a.model_dump())
                have.add(key)

    # Annotate with directory
    result["email_bundle"] = dpath.name
    result["source_directory"] = email_dir
    return result

def process_emails_concurrent(email_dirs: List[str]) -> List[Dict[str, Any]]:
    """Entry point for concurrent processing."""
    return asyncio.run(process_emails_concurrent_async(email_dirs))
def extract_bundle(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        first = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            response_format={"type":"json_object"},  # Use basic JSON format for compatibility
            **GEN_PARAMS
        )
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    provenance: List[Dict[str, Any]] = []
    tool_calls = first.choices[0].message.tool_calls or []

    # try direct JSON first
    if not tool_calls and (first.choices[0].message.content or "").strip():
        text = (first.choices[0].message.content or "").strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"):       text = text[:-3]
        try:
            payload = json.loads(text)
            obj = EmailEntityExtraction.model_validate(payload).model_dump()
            obj["provenance"] = provenance
            return obj
        except Exception as e:
            logger.warning(f"Direct JSON parse failed: {e}")
            pass

    follow = messages.copy()
    if tool_calls:
        follow.append({"role":"assistant","tool_calls": tool_calls})
        for tc in tool_calls:
            if tc.type == "function" and tc.function.name == "calculate_sha256":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    out  = calculate_sha256_local(args.get("content",""))
                    provenance.append({"tool_call_id": tc.id, **out})
                    follow.append({"role":"tool","tool_call_id": tc.id,"name": tc.function.name,"content": json.dumps(out)})
                except Exception as e:
                    follow.append({"role":"tool","tool_call_id": tc.id,"name": tc.function.name,"content": json.dumps({"error": str(e)})})

    try:
        final = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=follow + [{"role":"user","content":"Use any tool results above and return ONLY the JSON object."}],
            response_format={"type":"json_object"},  # Use basic JSON format for compatibility
            **GEN_PARAMS
        )
    except Exception as e:
        return {"error": f"API follow-up failed: {e}"}

    text = (final.choices[0].message.content or "").strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"):       text = text[:-3]
    try:
        payload = json.loads(text)
        obj = EmailEntityExtraction.model_validate(payload).model_dump()
        obj["provenance"] = provenance
        return obj
    except Exception as e:
        logger.error(f"validate/parse failed: {e}")
        return {"error": f"validate/parse failed: {e}", "raw_response": text[:500], "provenance": provenance}

# ──────────────────────────────────────────────────────────────────────────────
# Per-directory processing (one request) + merge attachments
# ──────────────────────────────────────────────────────────────────────────────
def process_email_directory_once(email_dir: str) -> Dict[str, Any]:
    dpath = Path(email_dir)
    modal = collect_modal_inputs_from_dir(dpath)
    messages = build_bundle_messages(modal["texts"], modal["image_paths"])
    result = extract_bundle(messages)

    if "error" not in result:
        result.setdefault("attachments", [])
        have = {(a.get("filename"), a.get("sha256")) for a in result["attachments"]}
        for a in modal["attachments"]:
            key = (a.filename, a.sha256)
            if key not in have:
                result["attachments"].append(a.model_dump())
                have.add(key)

    result["email_bundle"]     = dpath.name
    result["source_directory"] = email_dir
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Async over directories
# ──────────────────────────────────────────────────────────────────────────────
def get_already_processed_emails(case_id: str) -> Set[str]:
    case_out = Path(OUTPUT_DIR) / case_id
    jsonl = case_out / f"{case_id}_extractions.jsonl"
    processed: Set[str] = set()
    if jsonl.exists():
        with jsonl.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    if obj.get("email_bundle"): processed.add(obj["email_bundle"])
                except Exception:
                    continue
    return processed

async def _process_dirs_async(dirs: List[str]) -> List[Dict[str, Any]]:
    """Process all directories concurrently for maximum GPU utilization."""
    logger.info(f"Processing {len(dirs)} emails concurrently - vLLM handles batching across GPUs")
    return await asyncio.get_running_loop().run_in_executor(None, process_emails_concurrent, dirs)

def run_enhanced_email_ingestion(
    case_id: str,
    mapped_email_directory: str,
    max_emails: Optional[int] = None,
    batch_size: int = 100000,        # process all remaining by default
    use_async: bool = True
) -> Dict[str, Any]:
    """RAPIDS-optimized email ingestion with container networking"""

    base = Path(mapped_email_directory)
    if not base.exists():
        return {"status":"error","message":f"Directory not found: {base}"}

    case_out = Path(OUTPUT_DIR) / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    jsonl_file = case_out / f"{case_id}_extractions.jsonl"

    email_dirs = [str(p) for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    email_dirs.sort()
    if max_emails is not None:
        email_dirs = email_dirs[: max(0, max_emails)]

    processed = get_already_processed_emails(case_id)
    remaining = [d for d in email_dirs if Path(d).name not in processed]
    if not remaining:
        return {"status":"completed","message":"All emails already processed"}

    batch = remaining[:batch_size]
    logger.info(f"🚀 RAPIDS-optimized processing {len(batch)} directories (async={use_async})")
    logger.info(f"   Container networking: {OPENAI_BASE_URL}")
    logger.info(f"   GPU acceleration: {'✅ RAPIDS available' if RAPIDS_AVAILABLE else '❌ RAPIDS not available'}")

    start_time = datetime.now()

    if use_async:
        results = asyncio.run(_process_dirs_async(batch))
    else:
        results = [process_email_directory_once(d) for d in batch]

    processing_time = (datetime.now() - start_time).total_seconds()
    emails_per_second = len(results) / processing_time if processing_time > 0 else 0

    # RAPIDS-powered result analysis
    if RAPIDS_AVAILABLE and results:
        try:
            result_df = cudf.DataFrame({
                'has_error': [1 if 'error' in r else 0 for r in results],
                'has_nodes': [1 if 'nodes' in r and r.get('nodes') else 0 for r in results],
                'node_count': [len(r.get('nodes', [])) for r in results],
                'relationship_count': [len(r.get('relationships', [])) for r in results]
            })
            rapids_memory.track_dataframe(result_df)

            success_rate = (len(result_df) - result_df['has_error'].sum()) / len(result_df) * 100
            avg_nodes = result_df['node_count'].mean()
            avg_relationships = result_df['relationship_count'].mean()

            logger.info("📊 RAPIDS Result Analysis:")
            logger.info(f"   Success Rate: {success_rate:.1f}%")
            logger.info(f"   Average Nodes: {avg_nodes:.1f}")
            logger.info(f"   Average Relationships: {avg_relationships:.1f}")
        except Exception as e:
            logger.warning(f"RAPIDS result analysis failed: {e}")

    ok_count = err_count = 0
    with jsonl_file.open("a", encoding="utf-8") as f:
        for res in results:
            if isinstance(res, dict) and "error" not in res:
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")
                ok_count += 1
            else:
                err_count += 1

    # RAPIDS cleanup
    rapids_memory.cleanup()

    result = {
        "status": "success" if ok_count else "poor_quality",
        "case_id": case_id,
        "processed_emails": ok_count,
        "errors": err_count,
        "remaining_after_batch": len(remaining) - len(batch),
        "jsonl": str(jsonl_file),
        "processing_time_seconds": processing_time,
        "emails_per_second": emails_per_second,
        "rapids_accelerated": RAPIDS_AVAILABLE
    }

    logger.info("✅ RAPIDS email ingestion completed!")
    logger.info(f"   Performance: {emails_per_second:.2f} emails/second")
    logger.info(f"   Results: {ok_count}/{len(results)} successful")

    return result

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="RAPIDS-optimized multimodal email ingestion with Mistral vLLM")
    ap.add_argument("directory", nargs='+', help="Parent directory/directories containing email bundles (subdirectories)")
    ap.add_argument("--case-id", default="rapids_email_entity_extraction")
    ap.add_argument("--max-emails", type=int)
    ap.add_argument("--batch-size", type=int, default=100000)
    ap.add_argument("--sync", action="store_true", help="Disable async and run synchronously")
    args = ap.parse_args()

    # RAPIDS environment banner
    print("🚀 RAPIDS Email Ingestion with Mistral vLLM")
    print("=" * 50)
    print(f"📊 RAPIDS Available: {'✅ Yes' if RAPIDS_AVAILABLE else '❌ No'}")
    print(f"🌐 Mistral vLLM URL: {OPENAI_BASE_URL}")
    print(f"⚙️  Max Concurrent: {MAX_CONCURRENT_REQUESTS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print()

    # Collect all email directories from all provided parent directories
    all_email_dirs = []
    for parent_dir in args.directory:
        if not os.path.exists(parent_dir):
            print(f"❌ Directory {parent_dir} does not exist")
            return 1
        base = Path(parent_dir)
        email_dirs = [str(p) for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
        email_dirs.sort()
        all_email_dirs.extend(email_dirs)
        print(f"📁 Found {len(email_dirs)} email directories in {parent_dir}")

    print(f"📊 Total: {len(all_email_dirs)} email directories across {len(args.directory)} parent directories")
    print()

    # Run RAPIDS-optimized batch processing on all collected directories
    email_dirs_to_process = all_email_dirs[:args.max_emails] if args.max_emails else all_email_dirs
    logger.info(f"🚀 Starting RAPIDS-optimized batch processing of {len(email_dirs_to_process)} emails")
    logger.info(f"   Container networking: {OPENAI_BASE_URL}")
    logger.info(f"   GPU acceleration: {'✅ RAPIDS available' if RAPIDS_AVAILABLE else '❌ RAPIDS not available'}")

    start_time = datetime.now()
    results = process_emails_concurrent(email_dirs_to_process)
    end_time = datetime.now()

    processing_time = (end_time - start_time).total_seconds()
    successful_count = sum(1 for r in results if "error" not in r)
    emails_per_second = len(results) / processing_time if processing_time > 0 else 0

    # RAPIDS result analysis
    if RAPIDS_AVAILABLE and results:
        try:
            result_df = cudf.DataFrame({
                'has_error': [1 if 'error' in r else 0 for r in results],
                'has_nodes': [1 if 'nodes' in r and r.get('nodes') else 0 for r in results],
                'node_count': [len(r.get('nodes', [])) for r in results],
                'relationship_count': [len(r.get('relationships', [])) for r in results]
            })

            success_rate = (len(result_df) - result_df['has_error'].sum()) / len(result_df) * 100
            avg_nodes = result_df['node_count'].mean()
            avg_relationships = result_df['relationship_count'].mean()

            print("📊 RAPIDS Result Analysis:")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Average Nodes: {avg_nodes:.1f}")
            print(f"   Average Relationships: {avg_relationships:.1f}")
        except Exception as e:
            logger.warning(f"RAPIDS result analysis failed: {e}")

    logger.info(f"✨ Processing completed in {processing_time:.2f} seconds")
    logger.info(f"   Performance: {emails_per_second:.2f} emails/second")
    logger.info(f"   Results: {successful_count}/{len(results)} successful")

    # Save results to disk
    case_out = Path(OUTPUT_DIR) / args.case_id
    case_out.mkdir(parents=True, exist_ok=True)
    jsonl_file = case_out / f"{args.case_id}_extractions.jsonl"

    ok_count = err_count = 0
    with jsonl_file.open("w", encoding="utf-8") as f:
        for res in results:
            if isinstance(res, dict) and "error" not in res:
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")
                ok_count += 1
            else:
                # Still save error results for debugging
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")
                err_count += 1

    result = {
        "status": "success" if ok_count else "poor_quality",
        "case_id": args.case_id,
        "processed_emails": ok_count,
        "errors": err_count,
        "total_attempted": len(results),
        "jsonl": str(jsonl_file),
        "processing_time_seconds": processing_time,
        "emails_per_second": emails_per_second,
        "rapids_accelerated": RAPIDS_AVAILABLE
    }

    print()
    print("✅ RAPIDS Email Ingestion Complete!")
    print(f"📊 Performance: {emails_per_second:.2f} emails/second")
    print(f"🎯 Success Rate: {ok_count}/{len(results)} emails")
    print(f"📁 Output: {jsonl_file}")

    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
