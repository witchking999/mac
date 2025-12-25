import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dask.distributed import Client

from nemo_curator import Modify, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import NewlineNormalizer, UnicodeReformatter, UrlRemover
from nemo_curator.services import OpenAIClient
from nemo_curator.synthetic import NemotronCCGenerator
from openai import OpenAI


MESSAGE_TYPE = "MESSAGE"
DEFAULT_INPUT = "Eden_Complete_extractions.jsonl"
DEFAULT_CLEAN_OUTPUT = "processed_output.jsonl"
DEFAULT_QA_OUTPUT = "closed_qa_results.jsonl"
DEFAULT_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
DEFAULT_MODEL_KWARGS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 600,
}
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "docker"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def start_dask_client() -> Client:
    logging.info("Starting local Dask client for NeMo Curator operations.")
    return Client(processes=False, threads_per_worker=1, n_workers=1)


def extract_message_and_sha(row: Dict) -> Tuple[str, Optional[str]]:
    nodes = row.get("nodes", [])
    message_text = ""
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and node.get("type") == MESSAGE_TYPE:
                message_text = node.get("value", "")
                break

    message_sha = None
    hierarchy = row.get("email_sha_hierarchy")
    if isinstance(hierarchy, dict):
        for node_entry in hierarchy.get("node_shas", []):
            entity = node_entry.get("entity", {})
            if isinstance(entity, dict) and entity.get("type") == MESSAGE_TYPE:
                message_sha = node_entry.get("sha256")
                if message_sha:
                    break

    return message_text, message_sha


def load_dataset(input_path: Path) -> DocumentDataset:
    logging.info("Loading JSONL with DocumentDataset from %s", input_path)
    dataset = DocumentDataset.read_json(
        str(input_path),
        columns=[
            "case_id",
            "email_bundle",
            "nodes",
            "email_sha_hierarchy",
        ],
    )

    metadata_columns = [
        column
        for column in ("case_id", "email_bundle")
        if column in dataset.df.columns
    ]

    meta_frame = pd.DataFrame({col: pd.Series(dtype="object") for col in metadata_columns})
    meta_frame["text"] = pd.Series(dtype="object")
    meta_frame["message_sha256"] = pd.Series(dtype="object")

    def transform_partition(df: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict] = []
        for row in df.to_dict("records"):
            text, message_sha = extract_message_and_sha(row)
            if not text:
                continue
            record = {col: row.get(col) for col in metadata_columns}
            record["text"] = text
            record["message_sha256"] = message_sha
            records.append(record)
        return pd.DataFrame(records)

    transformed = dataset.df.map_partitions(transform_partition, meta=meta_frame)
    return DocumentDataset(transformed)


def build_processing_pipeline() -> Sequential:
    return Sequential(
        [
            Modify(UnicodeReformatter()),
            Modify(NewlineNormalizer()),
            Modify(UrlRemover()),
        ]
    )


def persist_clean_dataset(
    dataset: DocumentDataset,
    output_path: Path,
) -> DocumentDataset:
    logging.info("Cleaning documents with NeMo Curator pipeline.")
    pipeline = build_processing_pipeline()
    cleaned_dataset = pipeline(dataset).persist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing cleaned dataset to %s", output_path)
    cleaned_dataset.to_json(str(output_path), write_to_filename=False)
    return cleaned_dataset


def build_generator(base_url: str, api_key: str) -> NemotronCCGenerator:
    logging.info("Initialising Nemotron CC generator with base URL %s", base_url)
    openai_client = OpenAI(base_url=base_url, api_key=api_key)
    llm_client = OpenAIClient(openai_client)
    return NemotronCCGenerator(llm_client)


def parse_diverse_response(response_text: str) -> Iterable[Dict[str, Optional[str]]]:
    for line in response_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        stripped = stripped.lstrip("- ").strip()
        if "Question:" not in stripped:
            yield {"question": stripped, "answer": None}
            continue
        if "Answer:" in stripped:
            components = stripped.split("Question:", maxsplit=1)[1].split("Answer:", maxsplit=1)
            if len(components) == 2:
                question_part, answer_part = components
                yield {
                    "question": question_part.strip(),
                    "answer": answer_part.strip(),
                }
        else:
            yield {"question": stripped.split("Question:", maxsplit=1)[1].strip(), "answer": None}


def compute_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_qa_pairs(
    cleaned_dataset: DocumentDataset,
    generator: NemotronCCGenerator,
    model: str,
    model_kwargs: Dict,
    qa_output: Path,
    max_docs: Optional[int] = None,
) -> None:
    logging.info("Collecting cleaned documents for QA generation.")
    cleaned_df = cleaned_dataset.df.compute()
    if max_docs is not None:
        cleaned_df = cleaned_df.head(max_docs)

    qa_output.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Generating QA pairs for %s documents.", len(cleaned_df))
    with qa_output.open("w", encoding="utf-8") as handle:
        for _, row in cleaned_df.iterrows():
            document_text = row.get("text", "")
            if not document_text:
                continue

            responses = generator.generate_diverse_qa(
                document=document_text,
                model=model,
                model_kwargs=model_kwargs,
            )
            if not responses:
                continue

            for qa_pair in parse_diverse_response(responses[0]):
                record = {
                    "case_id": row.get("case_id"),
                    "email_bundle": row.get("email_bundle"),
                    "message_sha256": row.get("message_sha256"),
                    "cleaned_text_sha256": compute_sha256(document_text),
                    "question": qa_pair.get("question"),
                    "answer": qa_pair.get("answer"),
                    "model": model,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
