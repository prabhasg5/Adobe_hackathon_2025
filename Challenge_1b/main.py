import os
import sys
import json

# Add the Challenge_1a directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Challenge_1a'))

from processing.section_ranker import compute_similarity_scores
from processing.summarizer import summarize_sections
from processing.json_builder import build_final_output

# Import with correct function names
try:
    from src.extract_text import extract_text_with_metadata  # Fixed function name
    from src.heading_ranker import classify_headings         # Fixed function name
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

INPUT_PATH = "input/Challenge1b_input.json"
DOCS_DIR = "input/pdfs"
OUTPUT_PATH = "output/challenge1b_output.json"

def load_input():
    with open(INPUT_PATH, "r") as f:
        return json.load(f)

def process_documents(documents):
    doc_sections = []
    for doc in documents:
        file_path = os.path.join(DOCS_DIR, doc["filename"])
        text_blocks = extract_text_with_metadata(file_path)  # Fixed function name
        headings = classify_headings(text_blocks)            # Fixed function name
        doc_sections.append({
            "filename": doc["filename"],
            "title": doc.get("title", ""),
            "blocks": text_blocks,
            "headings": headings
        })
    return doc_sections

def transform_to_section_map(structured_docs):
    """Transform structured_docs to the format expected by compute_similarity_scores"""
    section_map = {}
    
    for doc in structured_docs:
        filename = doc['filename']
        sections = []
        
        # Convert headings to sections with content
        for i, heading in enumerate(doc['headings']):
            # Handle different heading formats
            if isinstance(heading, dict):
                title = heading.get('text', '')
                page_num = heading.get('page', 1)
            elif isinstance(heading, str):
                title = heading
                page_num = 1
            else:
                continue
            
            # Get content for this section
            content = ""
            if doc['blocks']:
                start_idx = i * 2
                end_idx = min(start_idx + 2, len(doc['blocks']))
                content = " ".join([block.get('text', '') for block in doc['blocks'][start_idx:end_idx]])
            
            sections.append({
                'title': title,
                'content': content,
                'page_number': page_num
            })
        
        # If no headings, create sections from text blocks
        if not sections and doc['blocks']:
            for i, block in enumerate(doc['blocks'][:5]):
                sections.append({
                    'title': f"Section {i+1}",
                    'content': block.get('text', ''),
                    'page_number': block.get('page', 1)
                })
        
        section_map[filename] = sections
    
    return section_map

def create_metadata(documents, structured_docs):
    """Create metadata dictionary for the final output"""
    metadata = {
        "total_documents": len(documents),
        "document_titles": [doc.get("title", doc["filename"]) for doc in documents],
        "processing_info": {
            "total_sections_processed": sum(len(doc.get('headings', [])) for doc in structured_docs),
            "documents_processed": len(structured_docs)
        }
    }
    return metadata

def main():
    os.makedirs("output", exist_ok=True)
    input_data = load_input()

    persona = input_data["persona"]
    job = input_data["job_to_be_done"]
    documents = input_data["documents"]

    print("[INFO] Extracting and structuring content from PDFs...")
    structured_docs = process_documents(documents)

    print("[INFO] Ranking sections by relevance to persona and task...")
    # Combine persona and job into single query
    query = f"Persona: {persona}. Job to be done: {job}"
    
    # Transform structured_docs to expected section_map format
    section_map = transform_to_section_map(structured_docs)
    
    # Call with correct arguments - use the imported function name
    ranked_sections = compute_similarity_scores(query, section_map)
    
    # Fix: Add the missing 'text' field to ranked_sections by matching with section_map
    for section in ranked_sections:
        document = section['document']
        section_title = section['section_title']
        
        # Find the corresponding content from section_map
        if document in section_map:
            for original_section in section_map[document]:
                if original_section['title'] == section_title:
                    section['text'] = original_section['content']
                    break
            else:
                section['text'] = ""  # Fallback if not found
        else:
            section['text'] = ""  # Fallback if document not found

    print("[INFO] Summarizing top-ranked sections...")
    summaries = summarize_sections(ranked_sections)

    print("[INFO] Building final output JSON...")
    # Create metadata for the build_final_output function
    metadata = create_metadata(documents, structured_docs)
    
    # Debug: Check what keys the ranked_sections actually have
    if ranked_sections:
        print("Sample ranked section keys:", list(ranked_sections[0].keys()))
        print("Sample ranked section:", ranked_sections[0])
    
    # Call with the correct parameter order: persona, job_to_be_done, extracted_sections, summarized_sections, metadata
    final_output = build_final_output(persona, job, ranked_sections, summaries, metadata)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"[SUCCESS] Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    