import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from agents import FunctionTool, RunContextWrapper
from pydantic import BaseModel, Field
from PIL import Image
import base64
from pdf2image import convert_from_path
import re
from datetime import datetime
from bs4 import BeautifulSoup

from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv("/home/witchking999/OTRTA/.env")

# API Configuration
QWEN_ENDPOINT_URL = "https://chwvwwr1m78j822j.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Initialize InferenceClient with endpoint URL (Inference Endpoints)
client = InferenceClient(model=QWEN_ENDPOINT_URL, token=HF_TOKEN)



OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/home/witchking999/sVi/solutionV-iNTELLiGENCE/api/output")
logger = logging.getLogger("enhanced_email_ingestion")

class EnhancedEmailIngestionInput(BaseModel):
    """Input schema for enhanced email ingestion"""
    case_id: str = Field(..., description="Case identifier for outputs")
    mapped_email_directory: str = Field(
        default="/home/witchking999/sVi/ingest_emails",
        description="Directory containing email directories to process"
    )
    max_emails: int = Field(default=5, description="Maximum number of emails to process")
    batch_size: int = Field(default=5, description="Number of emails to process in each batch")

# ============================================================================
# FILE CONVERSION FUNCTIONS
# ============================================================================

def convert_to_png(file_path):
    """Convert any image file to PNG format"""
    try:
        with Image.open(file_path) as img:
            png_path = os.path.splitext(file_path)[0] + '.png'
            img.save(png_path, 'PNG')
            return png_path
    except Exception as e:
        logger.error(f"Error converting {file_path} to PNG: {e}")
        return None

def convert_html_to_png(html_path):
    """Convert HTML file to PNG by extracting text and creating image"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin-1']
        html_content = None
        
        for encoding in encodings:
            try:
                with open(html_path, 'r', encoding=encoding) as f:
                    html_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if html_content is None:
            logger.error(f"Could not decode HTML file {html_path} with any encoding")
            return None
        
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        
        text_content = soup.get_text()
        img = Image.new('RGB', (800, 600), color='white')
        png_path = html_path.replace('.html', '_converted.png')
        img.save(png_path, 'PNG')
        return png_path
        
    except Exception as e:
        logger.error(f"Error converting HTML {html_path} to PNG: {e}")
        return None

def convert_pdf_to_png(pdf_path):
    """Convert PDF file to multiple PNG images (one per page)"""
    try:
        png_paths = []
        images = convert_from_path(pdf_path)
        
        for page_num, image in enumerate(images):
            png_path = f"{pdf_path.replace('.pdf', '')}_page_{page_num + 1}.png"
            image.save(png_path, 'PNG')
            png_paths.append(png_path)
        
        return png_paths
        
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path} to PNG: {e}")
        return []

# ============================================================================
# SHA PROVENANCE CALCULATION
# ============================================================================

def calculate_sha_provenance(content: str, content_type: str = "", source_file: str = "") -> Dict[str, Any]:
    """Calculate SHA256 hash for content provenance tracking"""
    try:
        sha256_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        provenance = {
            "content": content,
            "sha256": sha256_hash,
            "content_type": content_type,
            "source_file": source_file,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            "hash_algorithm": "SHA256"
        }
        
        logger.info(f"Generated SHA256 provenance for {content_type}: {sha256_hash[:16]}...")
        return provenance
        
    except Exception as e:
        logger.error(f"Error calculating SHA provenance: {e}")
        return {
            "content": content,
            "sha256": "",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# EMAIL PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_image(png_path: str) -> str:
    """Use chat completions with image input for OCR (endpoint-friendly)."""
    try:
        with open(png_path, 'rb') as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        prompt = (
            "Extract all visible text from this image. Return only the plain text, no extra narration."
        )

        response = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1000,
        )

        if hasattr(response, "choices") and response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""

    except Exception as e:
        logger.error(f"Error extracting text from {png_path}: {e}")
        return ""

def extract_structured_email_data(png_path: str, extracted_text: str) -> Dict[str, Any]:
    """Use chat completions with tool calls for structured extraction"""
    try:
        with open(png_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # System prompt for structured extraction
        system_prompt = (
            "You are an AI assistant specialized in email analysis and graph database construction. "
            "Analyze the email image and extract structured information into specific JSON format."
        )
        
        # User prompt with extracted text context
        user_prompt = (
            f"Based on this email image and the extracted text: '{extracted_text[:500]}...'\n\n"
            "TASK: Perform OCR on the image and extract email components for graph database construction:\n"
            "1. Email sender information\n"
            "2. Email recipients (to, cc, bcc)\n"
            "3. Subject line\n"
            "4. Date and time information\n"
            "5. Message body content\n"
            "6. Any attachments mentioned\n"
            "7. Any links or URLs\n\n"
            "INSTRUCTIONS:\n"
            "- Extract all visible text from the image\n"
            "- Identify email components and metadata\n"
            "- Use the calculate_sha256 tool to generate unique identifiers for extracted entities\n"
            "- Structure the output as JSON with 'nodes' and 'relationships' arrays\n\n"
            "NODES:\n"
            "- 'SENDER': Email sender information\n"
            "- 'RECIPIENT': Email recipient information\n"
            "- 'DATE': Date information from email\n"
            "- 'MESSAGE': Email body content\n"
            "- 'ATTACHMENT': File attachments\n"
            "- 'LINK': URLs and links mentioned\n\n"
            "RELATIONSHIPS:\n"
            "- 'HAS_SENDER': Connects MESSAGE to SENDER\n"
            "- 'HAS_RECIPIENT': Connects MESSAGE to RECIPIENT\n"
            "- 'HAS_SUBJECT': Connects MESSAGE to subject text\n"
            "- 'SENT_ON_DATE': Connects MESSAGE to DATE\n"
            "- 'HAS_ATTACHMENT': Connects MESSAGE to ATTACHMENT\n"
            "- 'HAS_LINK': Connects MESSAGE to LINK\n"
            "- 'MENTIONS_PERSON': Connects MESSAGE to mentioned persons\n\n"
            "Use the calculate_sha256 tool to generate SHA256 hashes for any extracted entities. Return the analysis results."
        )
        
        # Define the tool for SHA calculation
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_sha256",
                    "description": "Calculate SHA256 hash of content for provenance tracking",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to hash for provenance tracking",
                            },
                        },
                        "required": ["content"],
                    },
                },
            }
        ]

        # Structured Outputs schema (when provider supports grammar with tools)
        response_format = {
            "type": "json",
            "value": {
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "value": {"type": "string"},
                                "sha256": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["type", "value"],
                        },
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "sha256": {"type": "string"},
                            },
                            "required": ["type", "source", "target"],
                        },
                    },
                },
                "required": ["nodes", "relationships"],
            },
        }

        # Attempt with response_format + tools; fallback without if provider rejects
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_data}"},
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                tools=tools,
                tool_choice="auto",
                response_format=response_format,
                max_tokens=2000,
            )
        except Exception as e:
            # Known provider error: "Grammar and tools are mutually exclusive" -> retry without grammar
            logger.warning(f"response_format + tools failed ({e}); retrying without response_format")
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_data}"},
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=2000,
            )
        
        # Process tool calls for SHA calculation
        provenance_data = []
        tool_calls = getattr(response.choices[0].message, 'tool_calls', [])
        
        for tool_call in tool_calls:
            if tool_call.function.name == "calculate_sha256":
                try:
                    args = json.loads(tool_call.function.arguments)
                    content = args.get("content", "")
                    
                    sha_result = calculate_sha_provenance(
                        content=content,
                        content_type="extracted_entity",
                        source_file=png_path
                    )
                    provenance_data.append(sha_result)
                    logger.info(f"Tool call processed SHA for content: {sha_result['sha256'][:16]}...")
                    
                except Exception as e:
                    logger.error(f"Error processing SHA tool call: {e}")
        
        # Get response content
        result_text = response.choices[0].message.content.strip()
        logger.info(f"Received structured response: {result_text[:100]}...")
        
        # Parse JSON response
        try:
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            result = json.loads(result_text.strip())
            
            # Extract nodes and relationships from the graph structure
            all_nodes = []
            all_relationships = []
            
            # Process nodes from the response
            for node in result.get("nodes", []):
                if node.get("value"):  # Only process nodes with values
                    # Calculate SHA if not already present
                    if not node.get("sha256"):
                        node_sha = calculate_sha_provenance(
                            node["value"], 
                            node.get("type", "unknown"), 
                            png_path
                        )
                        node["sha256"] = node_sha["sha256"]
                        provenance_data.append(node_sha)
                    else:
                        # Create provenance entry for existing SHA
                        provenance_data.append(calculate_sha_provenance(
                            node["value"], 
                            node.get("type", "unknown"), 
                            png_path
                        ))
                    
                    # Ensure confidence is set
                    if "confidence" not in node:
                        node["confidence"] = 0.9
                    
                    all_nodes.append(node)
            
            # Process relationships from the response
            for rel in result.get("relationships", []):
                if rel.get("source") and rel.get("target"):  # Only process valid relationships
                    # Calculate SHA for relationship if not present
                    if not rel.get("sha256"):
                        rel_content = f"{rel['type']}:{rel['source']}->{rel['target']}"
                        rel_sha = calculate_sha_provenance(rel_content, "relationship", png_path)
                        rel["sha256"] = rel_sha["sha256"]
                        provenance_data.append(rel_sha)
                    
                    all_relationships.append(rel)
            
            logger.info(f"Successfully extracted: {len(all_nodes)} nodes, {len(provenance_data)} provenance entries")
            
            return {
                "nodes": all_nodes,
                "relationships": all_relationships,
                "provenance": provenance_data
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {result_text}")
            return {"nodes": [], "relationships": [], "provenance": provenance_data}
            
    except Exception as e:
        logger.error(f"Error in structured extraction for {png_path}: {e}")
        return {"nodes": [], "relationships": [], "provenance": []}

def perform_ner_on_png(png_path: str, email_context: str = "") -> Dict[str, Any]:
    """Main function that combines OCR and structured extraction"""
    try:
        logger.info(f"Processing image: {png_path}")
        
        # Step 1: Extract raw text using image_to_text
        extracted_text = extract_text_from_image(png_path)
        logger.info(f"Extracted text length: {len(extracted_text)} characters")
        
        # Step 2: Use structured extraction with tool calls
        result = extract_structured_email_data(png_path, extracted_text)
        
        return result
        
    except Exception as e:
        logger.error(f"Error performing NER on PNG {png_path}: {e}")
        return {"nodes": [], "relationships": [], "provenance": []}

def process_email_directory(email_dir: str, case_id: str) -> Dict[str, Any]:
    """Process all files in an email directory"""
    try:
        email_path = Path(email_dir)
        email_context = f"Email Directory: {email_path.name}"
        
        all_nodes = []
        all_relations = []
        all_provenance = []
        
        # Process all files
        for file_path in email_path.glob("*"):
            if not file_path.name.startswith('._') and file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                # Process HTML files
                if file_ext == '.html':
                    logger.info(f"Processing HTML file: {file_path}")
                    png_path = convert_html_to_png(str(file_path))
                    if png_path:
                        ner_result = perform_ner_on_png(png_path, email_context)
                        all_nodes.extend(ner_result.get("nodes", []))
                        all_relations.extend(ner_result.get("relationships", []))
                        all_provenance.extend(ner_result.get("provenance", []))
                
                # Process PDF files
                elif file_ext == '.pdf':
                    logger.info(f"Processing PDF file: {file_path}")
                    png_paths = convert_pdf_to_png(str(file_path))
                    for png_path in png_paths:
                        ner_result = perform_ner_on_png(png_path, email_context)
                        all_nodes.extend(ner_result.get("nodes", []))
                        all_relations.extend(ner_result.get("relationships", []))
                        all_provenance.extend(ner_result.get("provenance", []))
                
                # Process image files
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    if '_converted.png' in file_path.name:
                        continue  # Skip converted files
                    logger.info(f"Processing image file: {file_path}")
                    png_path = convert_to_png(str(file_path))
                    if png_path:
                        ner_result = perform_ner_on_png(png_path, email_context)
                        all_nodes.extend(ner_result.get("nodes", []))
                        all_relations.extend(ner_result.get("relationships", []))
                        all_provenance.extend(ner_result.get("provenance", []))
        
        # Add source information
        for entity in all_nodes:
            if isinstance(entity, dict):
                entity["source_document"] = email_dir
        
        for rel in all_relations:
            if isinstance(rel, dict):
                rel["source_document"] = email_dir
        
        # Validation
        senders = [e for e in all_nodes if e.get('type') == 'SENDER']
        recipients = [e for e in all_nodes if e.get('type') == 'RECIPIENT']
        
        logger.info(f"Processed {email_dir}: {len(all_nodes)} nodes, {len(all_relations)} relationships")
        logger.info(f"  - Senders: {[s.get('value') for s in senders]}")
        logger.info(f"  - Recipients: {[r.get('value') for r in recipients]}")
        
        # Save individual summary
        email_summary = {
            "nodes": all_nodes,
            "relationships": all_relations,
            "provenance": all_provenance,
            "validation": {
                "total_nodes": len(all_nodes),
                "total_relationships": len(all_relations),
                "total_provenance_entries": len(all_provenance),
                "extraction_quality": "good" if all_nodes else "poor"
            }
        }
        
        summary_path = os.path.join(OUTPUT_DIR, case_id, f"{email_path.name}/summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(email_summary, f, indent=2)
        
        return {"nodes": all_nodes, "relationships": all_relations, "provenance": all_provenance}
        
    except Exception as e:
        logger.error(f"Failed to process {email_dir}: {e}")
        return {"nodes": [], "relationships": [], "provenance": []}

def run_enhanced_email_ingestion(
    case_id: str,
    mapped_email_directory: str = "/home/witchking999/OTRTA/solv-iNTEL/ingest_emails",
    questions: List[str] = None,
    max_emails: int = 5,
    batch_size: int = 5
) -> Dict[str, Any]:
    """Enhanced email ingestion with dual approach: OCR + structured extraction"""
    logger.info(f"Starting enhanced email ingestion for case {case_id}")
    
    case_output = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_output, exist_ok=True)
    
    # Get email directories
    email_base_dir = Path(mapped_email_directory)
    if not email_base_dir.exists():
        return {"status": "error", "message": f"Email directory not found: {email_base_dir}"}
    
    email_dirs = [str(item) for item in email_base_dir.iterdir() 
                  if item.is_dir() and not item.name.startswith('.')]
    
    if not email_dirs:
        return {"status": "error", "message": "No email directories found"}
    
    # Limit to max_emails
    email_dirs = email_dirs[:max_emails]
    logger.info(f"Processing {len(email_dirs)} emails from {mapped_email_directory}")
    
    all_nodes = []
    all_relations = []
    all_provenance = []
    
    # Process each email directory
    for email_dir in email_dirs:
        result = process_email_directory(email_dir, case_id)
        all_nodes.extend(result["nodes"])
        all_relations.extend(result["relationships"])
        all_provenance.extend(result.get("provenance", []))
    
    # Validation
    all_senders = [e for e in all_nodes if e.get('type') == 'SENDER']
    all_recipients = [e for e in all_nodes if e.get('type') == 'RECIPIENT']
    
    logger.info(f"=== ENHANCED EMAIL INGESTION VALIDATION ===")
    logger.info(f"Total emails processed: {len(email_dirs)}")
    logger.info(f"Total nodes: {len(all_nodes)}")
    logger.info(f"Total relationships: {len(all_relations)}")
    logger.info(f"Senders found: {len(all_senders)}")
    logger.info(f"Recipients found: {len(all_recipients)}")
    
    status = "success" if all_nodes else "poor_quality"
    
    return {
        "status": status,
        "case_id": case_id,
        "processed_emails": len(email_dirs),
        "total_nodes": len(all_nodes),
        "total_relationships": len(all_relations),
        "total_provenance_entries": len(all_provenance),
        "email_directories": email_dirs,
        "validation": {
            "senders_found": len(all_senders),
            "recipients_found": len(all_recipients),
            "overall_quality": "good" if status == "success" else "poor"
        },
        "provenance_summary": {
            "total_sha_hashes": len(set(p.get("sha256") for p in all_provenance if p.get("sha256"))),
            "content_types": list(set(p.get("content_type") for p in all_provenance if p.get("content_type"))),
            "sample_provenance": all_provenance[:5] if all_provenance else []
        }
    }

# Tool definition
enhanced_email_ingestion_tool = FunctionTool(
    name="EnhancedEmailIngestion",
    description="Enhanced email ingestion with dual approach: OCR + structured extraction with tool calls",
    params_json_schema=EnhancedEmailIngestionInput.model_json_schema(),
    on_invoke_tool=RunContextWrapper(run_enhanced_email_ingestion)
)

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_enhanced_email_ingestion():
    """Test the enhanced email ingestion tool"""
    case_id = "test_dual_approach"
    test_email_directory = "/home/witchking999/OTRTA/solv-iNTEL/ingest_emails"
    
    result = run_enhanced_email_ingestion(
        case_id=case_id,
        mapped_email_directory=test_email_directory,
        max_emails=3
    )
    
    # Save test results
    test_output_file = os.path.join(OUTPUT_DIR, case_id, "test_results.json")
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)
    
    with open(test_output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test results saved to: {test_output_file}")
    logger.info(f"Test Summary: {result['total_nodes']} nodes, {result['total_relationships']} relationships, {result['total_provenance_entries']} provenance entries")
    
    return result

if __name__ == "__main__":
    test_enhanced_email_ingestion()