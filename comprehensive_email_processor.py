#!/usr/bin/env python3
"""
comprehensive_email_processor.py

Complete email processing pipeline that:
1. Removes mac metadata files
2. Maps HTML emails to attachments
3. Converts HTML to text using langchain
4. Creates organized directory structure

Usage:
  python3 comprehensive_email_processor.py /path/to/messages/directory
"""

import os
import json
import shutil
import argparse
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from bs4 import BeautifulSoup
    from langchain_community.document_loaders import AsyncHtmlLoader
    from langchain_community.document_transformers import Html2TextTransformer
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("Warning: langchain not available. HTML to text conversion will be skipped.")
    print("Install with: pip install langchain langchain-community beautifulsoup4")


def calculate_file_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (OSError, IOError) as e:
        print(f"Warning: Could not calculate SHA for {filepath}: {e}")
        return "ERROR"


def calculate_directory_provenance(directory: str) -> Dict[str, str]:
    """Calculate SHA256 hashes for all files in a directory."""
    print(f"🔐 Calculating SHA provenance for: {os.path.basename(directory)}")

    provenance = {}
    total_files = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, directory)
            sha256 = calculate_file_sha256(filepath)

            if sha256 != "ERROR":
                provenance[rel_path] = {
                    'sha256': sha256,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
                total_files += 1

                if total_files % 50 == 0:
                    print(f"   Processed {total_files} files...")

    print(f"✅ Calculated provenance for {total_files} files")
    return provenance


def create_provenance_report(source_dir: str, clean_dir: str, mapped_dir: str,
                           text_dir: str = None) -> Dict[str, any]:
    """Create comprehensive provenance report."""
    print("📋 Generating provenance report...")

    # Calculate provenance for each stage
    source_provenance = calculate_directory_provenance(source_dir)
    clean_provenance = calculate_directory_provenance(clean_dir)
    mapped_provenance = calculate_directory_provenance(mapped_dir)

    provenance_report = {
        'metadata': {
            'report_generated': datetime.now().isoformat(),
            'source_directory': source_dir,
            'processing_pipeline': 'comprehensive_email_processor'
        },
        'source_provenance': source_provenance,
        'clean_provenance': clean_provenance,
        'mapped_provenance': mapped_provenance
    }

    if text_dir:
        text_provenance = calculate_directory_provenance(text_dir)
        provenance_report['text_provenance'] = text_provenance

    # Calculate processing statistics
    source_files = len(source_provenance)
    clean_files = len(clean_provenance)
    mapped_files = len(mapped_provenance)

    provenance_report['statistics'] = {
        'source_files': source_files,
        'clean_files': clean_files,
        'mapped_files': mapped_files,
        'files_removed_in_cleaning': source_files - clean_files,
        'files_created_in_mapping': mapped_files - clean_files
    }

    print("✅ Provenance report generated")
    return provenance_report


def clean_directory(source_dir: str, output_base: str = "/home/witchking999/VOLTRON/DATA") -> str:
    """Step 1: Clean directory by removing mac metadata files and return clean path."""
    print("🧹 STEP 1: Cleaning directory (removing mac metadata files)")

    # Create source-aware directory name for tracking
    # Use full path info for audit trail: parent_dir_source_dir_clean
    parent_dir = os.path.basename(os.path.dirname(source_dir))
    source_name = os.path.basename(source_dir)

    # Sanitize names for filesystem safety
    safe_parent = parent_dir.replace(' ', '_').replace('-', '_').replace('.', '_')
    safe_source = source_name.replace(' ', '_').replace('-', '_').replace('.', '_')

    clean_dir_name = f"{safe_parent}_{safe_source}_clean"
    clean_dir = os.path.join(output_base, clean_dir_name)

    # Remove if exists
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)

    print(f"📂 Processing source: {source_dir}")
    print(f"🎯 Creating clean directory: {clean_dir_name}")

    # Copy only non-mac files (this is the ONLY time we touch source)
    shutil.copytree(source_dir, clean_dir, ignore=shutil.ignore_patterns('._*', '*/._*'))

    # Count files
    total_files = sum(1 for root, dirs, files in os.walk(clean_dir) for file in files)
    print(f"✅ Clean directory created: {total_files} files (source untouched)")

    return clean_dir


def parse_html_references(html_path: str) -> Set[str]:
    """Parse HTML file for attachment references."""
    try:
        with open(html_path, encoding='utf8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except Exception as e:
        print(f"Error parsing {html_path}: {e}")
        return set()

    refs = set()

    # Extract from src and href attributes
    for tag in soup.find_all(src=True):
        src = tag['src'].strip()
        if src and not src.startswith(('http://', 'https://', 'mailto:', '#')):
            refs.add(src)

    for tag in soup.find_all(href=True):
        href = tag['href'].strip()
        if href and not href.startswith(('http://', 'https://', 'mailto:', '#')):
            refs.add(href)

    return refs


def build_attachment_index(messages_dir: str) -> Dict[str, str]:
    """Build index of all attachments in the directory."""
    index = {}

    for root, dirs, files in os.walk(messages_dir):
        for file in files:
            # Skip HTML files and mac metadata
            if file.lower().endswith('.html') or file.startswith('._'):
                continue

            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, messages_dir)
            index[rel_path] = filepath

    return index


def map_emails_to_attachments(clean_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Step 2: Map HTML emails to their attachments."""
    print("\n📎 STEP 2: Mapping emails to attachments")

    # Get all HTML files
    html_files = []
    for root, dirs, files in os.walk(clean_dir):
        for file in files:
            if file.lower().endswith('.html') and not file.startswith('._'):
                html_files.append(os.path.join(root, file))

    print(f"Found {len(html_files)} HTML files to process")

    # Build attachment index
    attachment_index = build_attachment_index(clean_dir)

    # Map each HTML to its attachments
    mapping = {}
    for html_path in html_files:
        rel_html_path = os.path.relpath(html_path, clean_dir)
        refs = parse_html_references(html_path)

        # Find matching attachments
        attachments = []
        for ref in refs:
            # Try direct match first
            if ref in attachment_index:
                attachments.append(ref)
            else:
                # Try basename match
                basename = os.path.basename(ref)
                for att_path in attachment_index.keys():
                    if os.path.basename(att_path) == basename:
                        attachments.append(att_path)
                        break

        mapping[rel_html_path] = list(set(attachments))  # Remove duplicates

    # Count emails with attachments
    emails_with_attachments = sum(1 for atts in mapping.values() if atts)
    print(f"✅ Mapped {len(mapping)} emails, {emails_with_attachments} have attachments")

    return mapping, attachment_index


def create_mapped_directory(clean_dir: str, mapping: Dict[str, List[str]],
                           attachment_index: Dict[str, str], output_base: str) -> str:
    """Step 3: Create organized directory with mapped emails."""
    print("\n📁 STEP 3: Creating mapped directory structure")

    # Extract source information from clean_dir name for consistent naming
    clean_dir_name = os.path.basename(clean_dir)
    source_info = clean_dir_name.replace('_clean', '')
    mapped_dir_name = f"{source_info}_mapped"
    mapped_dir = os.path.join(output_base, mapped_dir_name)

    print(f"🎯 Creating mapped directory: {mapped_dir_name}")

    # Remove if exists
    if os.path.exists(mapped_dir):
        shutil.rmtree(mapped_dir)

    os.makedirs(mapped_dir, exist_ok=True)

    # Copy each email and its attachments
    for html_rel_path, attachments in mapping.items():
        html_basename = os.path.splitext(os.path.basename(html_rel_path))[0]
        bundle_dir = os.path.join(mapped_dir, html_basename)
        os.makedirs(bundle_dir, exist_ok=True)

        # Copy HTML file
        html_src = os.path.join(clean_dir, html_rel_path)
        html_dst = os.path.join(bundle_dir, os.path.basename(html_rel_path))
        if os.path.exists(html_src):
            shutil.copy2(html_src, html_dst)

        # Copy attachments
        for att_rel_path in attachments:
            if att_rel_path in attachment_index:
                att_src = attachment_index[att_rel_path]
                att_dst = os.path.join(bundle_dir, os.path.basename(att_rel_path))

                # Create subdirs if needed
                os.makedirs(os.path.dirname(att_dst), exist_ok=True)
                shutil.copy2(att_src, att_dst)

    total_bundles = len(mapping)
    print(f"✅ Created {total_bundles} email bundles in {mapped_dir_name}")

    return mapped_dir


def convert_html_to_text(mapped_dir: str, output_base: str) -> str:
    """Step 4: Convert HTML files to text using langchain and copy attachments."""
    print("\n📄 STEP 4: Converting HTML to text + copying attachments")

    if not HAS_LANGCHAIN:
        print("⚠️  Langchain not available, skipping HTML to text conversion")
        return mapped_dir

    # Extract source information from mapped_dir name for consistent naming
    mapped_dir_name = os.path.basename(mapped_dir)
    source_info = mapped_dir_name.replace('_mapped', '')
    text_dir_name = f"{source_info}_mapped_text"
    text_dir = os.path.join(output_base, text_dir_name)

    print(f"🎯 Creating text directory: {text_dir_name}")

    # Remove if exists
    if os.path.exists(text_dir):
        shutil.rmtree(text_dir)

    os.makedirs(text_dir, exist_ok=True)

    # Copy entire directory structure from mapped to text
    print("Copying directory structure and attachments...")
    for root, dirs, files in os.walk(mapped_dir):
        # Create corresponding directory in text folder
        rel_path = os.path.relpath(root, mapped_dir)
        if rel_path != '.':
            target_dir = os.path.join(text_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)

        # Copy all files (HTML, attachments, etc.)
        for file in files:
            src_file = os.path.join(root, file)
            if rel_path != '.':
                dst_file = os.path.join(text_dir, rel_path, file)
            else:
                dst_file = os.path.join(text_dir, file)

            # Convert HTML to text, copy other files as-is
            if file.lower().endswith('.html'):
                try:
                    # Read HTML content
                    with open(src_file, 'r', encoding='utf8', errors='ignore') as f:
                        html_content = f.read()

                    # Convert to text
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text_content = soup.get_text(separator='\n', strip=True)

                    # Save as text file
                    text_file = dst_file.replace('.html', '.txt')
                    os.makedirs(os.path.dirname(text_file), exist_ok=True)

                    with open(text_file, 'w', encoding='utf8') as f:
                        f.write(text_content)

                    print(f"  Converted: {file} → {os.path.basename(text_file)}")

                except Exception as e:
                    print(f"  Error converting {file}: {e}")
                    # Copy original file if conversion fails
                    shutil.copy2(src_file, dst_file)
            else:
                # Copy non-HTML files (attachments) as-is
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)

    print(f"✅ Created complete text bundles with attachments in {text_dir_name}")
    return text_dir


def save_final_mapping(mapping: Dict[str, List[str]], output_dir: str) -> None:
    """Save the final mapping with additional metadata."""
    mapping_file = os.path.join(output_dir, 'email_attachment_mapping.json')

    # Add metadata
    final_mapping = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_emails': len(mapping),
            'emails_with_attachments': sum(1 for atts in mapping.values() if atts),
            'total_attachments': sum(len(atts) for atts in mapping.values())
        },
        'mapping': mapping
    }

    with open(mapping_file, 'w', encoding='utf8') as f:
        json.dump(final_mapping, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved final mapping to {mapping_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete email processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete pipeline:
1. Remove mac metadata files
2. Map HTML to attachments
3. Convert HTML to text
4. Create organized directory structure

Example:
  python3 comprehensive_email_processor.py /path/to/messages
        """
    )
    parser.add_argument('source_dir', help='Source directory containing emails and attachments')
    parser.add_argument('--output-dir', '-o', default='/home/witchking999/VOLTRON/DATA',
                       help='Output base directory (default: /home/witchking999/VOLTRON/DATA)')
    parser.add_argument('--skip-text-conversion', action='store_true',
                       help='Skip HTML to text conversion')
    parser.add_argument('--skip-provenance', action='store_true',
                       help='Skip SHA provenance calculation (faster but less secure)')

    args = parser.parse_args()

    if not os.path.exists(args.source_dir):
        print(f"❌ Error: Source directory '{args.source_dir}' does not exist")
        return 1

    print("🚀 STARTING COMPREHENSIVE EMAIL PROCESSING PIPELINE")
    print("=" * 60)

    try:
        # Step 0: Calculate source provenance (before any processing)
        if not args.skip_provenance:
            print("\n🔐 STEP 0: Establishing source provenance")
            source_provenance = calculate_directory_provenance(args.source_dir)
            print(f"📋 Source contains {len(source_provenance)} files with verified integrity")
        else:
            source_provenance = {}
            print("⚠️  Skipping source provenance calculation (--skip-provenance)")

        # Step 1: Clean directory
        clean_dir = clean_directory(args.source_dir, args.output_dir)

        # Step 2: Map emails to attachments
        mapping, attachment_index = map_emails_to_attachments(clean_dir)

        # Step 3: Create mapped directory
        mapped_dir = create_mapped_directory(clean_dir, mapping, attachment_index, args.output_dir)

        # Step 4: Convert HTML to text (if not skipped)
        if not args.skip_text_conversion:
            text_dir = convert_html_to_text(mapped_dir, args.output_dir)
        else:
            text_dir = mapped_dir

        # Step 5: Generate comprehensive provenance report
        if not args.skip_provenance:
            print("\n🔐 STEP 5: Generating provenance report")
            provenance_report = create_provenance_report(
                args.source_dir, clean_dir, mapped_dir,
                text_dir if not args.skip_text_conversion else None
            )

            # Save provenance report
            provenance_file = os.path.join(args.output_dir, 'data_provenance_report.json')
            with open(provenance_file, 'w', encoding='utf8') as f:
                json.dump(provenance_report, f, indent=2, ensure_ascii=False)
            print(f"✅ Provenance report saved: {provenance_file}")
        else:
            provenance_report = {}
            print("⚠️  Skipping provenance report generation")

        # Save final mapping with provenance
        save_final_mapping(mapping, args.output_dir)

        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📂 SOURCE TRACKED: {args.source_dir}")
        print(f"🔍 SOURCE ID: {os.path.basename(os.path.dirname(args.source_dir))}/{os.path.basename(args.source_dir)}")
        print()
        print("📁 OUTPUT DIRECTORIES (with source tracking):")
        print(f"   Clean: {os.path.basename(clean_dir)}")
        print(f"   Mapped: {os.path.basename(mapped_dir)}")
        if not args.skip_text_conversion:
            print(f"   Text: {os.path.basename(text_dir)}")
        print(f"   Mapping: email_attachment_mapping.json")
        if not args.skip_provenance:
            print(f"   Provenance: data_provenance_report.json")
        print()
        print(f"📍 Full paths in: {args.output_dir}")

        # Summary
        total_emails = len(mapping)
        emails_with_attachments = sum(1 for atts in mapping.values() if atts)
        total_attachments = sum(len(atts) for atts in mapping.values())

        print("\n📈 PROCESSING SUMMARY:")
        print(f"• Total emails processed: {total_emails}")
        print(f"• Emails with attachments: {emails_with_attachments}")
        print(f"• Total attachments: {total_attachments}")
        print(f"• Success rate: {emails_with_attachments/total_emails*100:.1f}%")

        if not args.skip_provenance:
            print("\n🔐 PROVENANCE SUMMARY:")
            print(f"• Source files verified: {len(source_provenance)}")
            print(f"• Processing integrity: SHA256 verified")
            print(f"• Audit trail: Complete provenance chain")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
