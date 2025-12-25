from nemo_curator import Sequential, Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import UnicodeReformatter, UrlRemover, NewlineNormalizer
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.services import OpenAIClient
from nemo_curator.synthetic import NemotronCCGenerator

from openai import OpenAI  # If using local vLLM or NVIDIA endpoint
import json
import os
import pandas as pd
from dask.distributed import Client
import yaml

from dask.distributed import Client

# Start a local Dask client for NeMo Curator
client = Client(processes=False, threads_per_worker=1, n_workers=1)

# --- Configurable parameters ---
input_path = "Eden_Complete_extractions.jsonl"       # Path to the Eden Complete extractions JSONL
output_cleaned = "processed_output.jsonl"    # File for cleaned data
output_qa = "closed_qa_results.jsonl"   # QA output file

base_url = "http://localhost:8000/v1"  # Local vLLM endpoint for Mistral
api_key = "docker"  # API key for local endpoint

model = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
n_openlines = 5

model_kwargs = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 600,
}

# --- Load and preprocess documents from Eden Complete extractions ---
print("Loading data from Eden Complete extractions...")
documents = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        # Extract the MESSAGE text from nodes
        message_text = None
        for node in data.get("nodes", []):
            if node.get("type") == "MESSAGE":
                message_text = node.get("value", "")
                break
        if message_text:
            documents.append({"text": message_text})

print(f"Loaded {len(documents)} documents from Eden Complete extractions.")

# Create DocumentDataset from the list of dicts
dataset = DocumentDataset.from_pandas(pd.DataFrame(documents))

processing_pipeline = Sequential([
    Modify(UnicodeReformatter()),
    Modify(NewlineNormalizer()),
    Modify(UrlRemover()),
    # Modify(PiiModifier(
    #     language="en",
    #     supported_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    #     anonymize_action="redact"
    # ))
])

cleaned_dataset = processing_pipeline(dataset)
cleaned_dataset.to_json(output_cleaned, write_to_filename=False)

# --- Extract cleaned text for QA generation ---
documents = cleaned_dataset.df["text"].compute().tolist()

print(f"Cleaned {len(documents)} documents... Now generating closed Q&A pairs.")

# --- Setup LLM Client and pipeline for Q&A generation ---
openai_client = OpenAI(
    base_url=base_url,
    api_key=api_key
)
llm_client = OpenAIClient(openai_client)
generator = NemotronCCGenerator(llm_client)

# --- Generate Q&A pairs for every cleaned document ---
results = []
for i, doc in enumerate(documents):
    responses = generator.generate_diverse_qa(
        document=doc,
        model=model,
        model_kwargs=model_kwargs
    )

    # Parse the response to extract QA pairs
    response_text = responses[0]
    # Assume the format is "Here are the questions and answers...\n- Question: ... Answer: ..."
    lines = response_text.split('\n')
    qa_pairs = []
    for line in lines:
        if line.strip().startswith('- Question:') or line.strip().startswith('- '):
            # Parse the line
            if 'Question:' in line and 'Answer:' in line:
                parts = line.split('Question:')[1].split('Answer:')
                if len(parts) == 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                    qa_pairs.append({"question": question, "answer": answer})
            elif 'Question:' in line:
                # Only question
                question = line.split('Question:')[1].strip()
                qa_pairs.append({"question": question, "answer": None})
            else:
                # Just the text as question
                qa_pairs.append({"question": line.strip().lstrip('- ').strip(), "answer": None})

    for qa in qa_pairs:
        if isinstance(qa, dict):
            results.append({
                "doc_index": i,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "text_excerpt": doc[:128]  # For provenance/debug, optional
            })
        else:
            results.append({
                "doc_index": i,
                "question": qa,
                "answer": None,
                "text_excerpt": doc[:128]
            })

# --- Save full Q&A results to file ---
with open(output_qa, "w", encoding="utf-8") as out_f:
    for entry in results:
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Q&A results saved to {output_qa}")
 