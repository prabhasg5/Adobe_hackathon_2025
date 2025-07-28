import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Challenge_1a'))

from processing.section_ranker import compute_similarity_scores
from processing.summarizer import summarize_sections
from processing.json_builder import build_final_output
from processing.heading_ranker import classify_headings, classify_headings_with_content
from processing.extract_text import extract_text_with_metadata
INPUT_PATH = "input/Challenge1b_input.json"
DOCS_DIR = "input/pdfs"
OUTPUT_PATH = "output/challenge1b_output.json"

def load_input():
    with open(INPUT_PATH, "r") as f:
        return json.load(f)

def process_documents(documents: List[Dict]) -> List[Dict]:
    doc_sections = []
    
    for doc in documents:
        file_path = os.path.join(DOCS_DIR, doc["filename"])
        
        if not os.path.exists(file_path):
            continue
        
        try:
            text_blocks = extract_text_with_metadata(file_path)
            headings, sections = classify_headings_with_content(text_blocks)
            
            doc_data = {
                "filename": doc["filename"],
                "title": doc.get("title", doc["filename"]),
                "blocks": text_blocks,
                "headings": headings,
                "sections": sections,
                "document_stats": {
                    "total_blocks": len(text_blocks),
                    "total_headings": len(headings),
                    "total_sections": len(sections),
                    "pages": max([block.get("page", 1) for block in text_blocks], default=1)
                }
            }
            
            doc_sections.append(doc_data)
            
        except Exception as e:
            continue
    
    return doc_sections

def create_comprehensive_section_map(structured_docs: List[Dict]) -> Dict[str, List[Dict]]:
    """Create section map prioritizing high-level comprehensive sections."""
    section_map = {}
    
    for doc in structured_docs:
        filename = doc['filename']
        sections = []
        
        if doc.get('sections'):
            # Filter and prioritize sections
            filtered_sections = filter_and_prioritize_sections(doc['sections'], doc['headings'])
            
            for section in filtered_sections:
                sections.append({
                    'title': section['title'],
                    'content': section['content'] or section['title'],  # Fallback to title if no content
                    'page_number': section['page_number'],
                    'level': section.get('level', 'H1')
                })
        elif doc.get('headings'):
            sections = create_sections_from_headings(doc['headings'], doc['blocks'])
        else:
            sections = create_sections_from_blocks(doc['blocks'])
        
        section_map[filename] = sections
    
    return section_map

def filter_and_prioritize_sections(sections: List[Dict], headings: List[Dict]) -> List[Dict]:
    """Filter sections to prioritize comprehensive, high-level content."""
    if not sections:
        return sections
    
    # Prioritize H1 and H2 sections (main topics)
    priority_sections = []
    supplementary_sections = []
    
    for section in sections:
        level = section.get('level', 'H1')
        title = section.get('title', '').lower()
        content = section.get('content', '')
        
        # Calculate section quality score
        quality_score = calculate_section_quality(section)
        
        # Prioritize based on level and quality
        if level in ['H1', 'H2'] and quality_score > 3.0:
            priority_sections.append(section)
        elif quality_score > 2.0:
            supplementary_sections.append(section)
    
    # Sort priority sections by quality
    priority_sections.sort(key=lambda x: calculate_section_quality(x), reverse=True)
    supplementary_sections.sort(key=lambda x: calculate_section_quality(x), reverse=True)
    
    # Return top priority sections + some supplementary ones
    return priority_sections[:8] + supplementary_sections[:4]

def calculate_section_quality(section: Dict) -> float:
    """Calculate quality score for a section to prioritize comprehensive content."""
    title = section.get('title', '').lower()
    content = section.get('content', '')
    level = section.get('level', 'H1')
    
    score = 0.0
    
    # Level importance (H1 > H2 > H3)
    if level == 'H1':
        score += 3.0
    elif level == 'H2':
        score += 2.0
    elif level == 'H3':
        score += 1.0
    
    # Title quality indicators (comprehensive topics)
    comprehensive_indicators = [
        'comprehensive', 'guide', 'overview', 'introduction', 'major', 'main',
        'complete', 'full', 'total', 'entire', 'all', 'general'
    ]
    
    for indicator in comprehensive_indicators:
        if indicator in title:
            score += 2.0
            break
    
    # Content length (substantial content is often more comprehensive)
    content_length = len(content)
    if content_length > 200:
        score += 1.5
    elif content_length > 100:
        score += 1.0
    elif content_length > 50:
        score += 0.5
    
    # Title length (not too short, not too long)
    title_words = len(title.split())
    if 2 <= title_words <= 8:
        score += 1.0
    elif title_words == 1:
        score += 0.5
    
    # Penalize overly specific/detailed topics
    detail_indicators = [
        'tip', 'trick', 'specific', 'detail', 'particular', 'individual',
        'personal', 'private', 'small', 'minor', 'single'
    ]
    
    for indicator in detail_indicators:
        if indicator in title:
            score -= 1.0
            break
    
    return max(0.0, score)

def create_sections_from_headings(headings: List[Dict], blocks: List[Dict]) -> List[Dict]:
    """Create sections from headings, prioritizing high-level ones."""
    # Filter headings to focus on main topics
    filtered_headings = []
    
    for heading in headings:
        level = heading.get('level', 'H1')
        likelihood = heading.get('likelihood', 0)
        
        # Prioritize H1 and high-likelihood H2 headings
        if level == 'H1' or (level == 'H2' and likelihood > 3.0):
            filtered_headings.append(heading)
    
    # If we don't have enough high-level headings, add some H2s
    if len(filtered_headings) < 3:
        for heading in headings:
            if heading.get('level') == 'H2' and heading not in filtered_headings:
                filtered_headings.append(heading)
                if len(filtered_headings) >= 6:  # Reasonable limit
                    break
    
    sections = []
    sorted_blocks = sorted(blocks, key=lambda x: (x.get("page", 1), x.get("y", 0)))
    
    for i, heading in enumerate(filtered_headings):
        heading_page = heading["page"]
        heading_position = heading.get("position", 0)
        content_blocks = []
        
        if i < len(filtered_headings) - 1:
            next_heading = filtered_headings[i + 1]
            next_page = next_heading["page"]
            next_position = next_heading.get("position", float('inf'))
        else:
            next_page = float('inf')
            next_position = float('inf')
        
        for block in sorted_blocks:
            block_page = block.get("page", 1)
            block_position = block.get("y", 0)
            block_text = block.get("text", "").strip()
            
            if block_text == heading["text"]:
                continue
            
            in_section = False
            
            if block_page == heading_page and block_position > heading_position:
                if next_page > heading_page or (next_page == heading_page and block_position < next_position):
                    in_section = True
            elif block_page > heading_page:
                if block_page < next_page or (block_page == next_page and block_position < next_position):
                    in_section = True
            
            if in_section:
                if not is_likely_another_heading(block, filtered_headings):
                    content_blocks.append(block_text)
        
        # Take more content for comprehensive sections
        content = " ".join(content_blocks[:10]).strip()  # Increased from 5 to 10
        
        sections.append({
            'title': heading["text"],
            'content': content if content else heading["text"],
            'page_number': heading["page"],
            'level': heading.get("level", "H1")
        })
    
    return sections

def create_sections_from_blocks(blocks: List[Dict]) -> List[Dict]:
    """Create sections from high-scoring text blocks when no headings are found."""
    sections = []
    
    # Filter blocks that could serve as section titles
    title_candidates = []
    for block in blocks:
        # Use heading score if available
        score = block.get("heading_score", 0)
        if score == 0:
            # Calculate basic score
            score = calculate_block_heading_score(block)
        
        if score > 1.5:  # Lower threshold for fallback mode
            title_candidates.append((block, score))
    
    # Sort by score and take top candidates
    title_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Create sections from top candidates
    for i, (block, score) in enumerate(title_candidates[:5]):  # Max 5 sections
        content = block.get("text", "")
        
        # Try to find some additional content
        page = block.get("page", 1)
        position = block.get("y", 0)
        
        additional_content = []
        for other_block in blocks:
            if (other_block.get("page", 1) == page and 
                other_block.get("y", 0) > position and
                other_block != block):
                additional_content.append(other_block.get("text", ""))
                if len(additional_content) >= 5:  # Increased content
                    break
        
        if additional_content:
            content += " " + " ".join(additional_content)
        
        sections.append({
            'title': block.get("text", f"Section {i+1}")[:50],  # Limit title length
            'content': content,
            'page_number': page,
            'level': "H1" if i == 0 else "H2"
        })
    
    # If no good candidates, create basic sections from first few blocks
    if not sections:
        for i, block in enumerate(blocks[:3]):
            sections.append({
                'title': f"Section {i+1}",
                'content': block.get("text", ""),
                'page_number': block.get("page", 1),
                'level': "H1" if i == 0 else "H2"
            })
    
    return sections

def is_likely_another_heading(block: Dict, all_headings: List[Dict]) -> bool:
    """Check if a block is likely another heading that should not be included as content."""
    block_text = block.get("text", "").strip()
    
    # Check if this text matches any known heading
    for heading in all_headings:
        if heading["text"] == block_text:
            return True
    
    # Check if it has heading-like characteristics
    score = block.get("heading_score", 0)
    if score == 0:
        score = calculate_block_heading_score(block)
    
    return score > 2.0

def calculate_block_heading_score(block: Dict) -> float:
    """Calculate a basic heading score for a block."""
    text = block.get("text", "")
    font_size = block.get("font_size", 12)
    word_count = len(text.split())
    
    score = 0.0
    
    # Font size (assume 12 is average)
    if font_size > 14:
        score += 2.0
    elif font_size > 12:
        score += 1.0
    
    # Length characteristics
    if 1 <= word_count <= 8:
        score += 1.5
    elif word_count > 15:
        score -= 1.0
    
    # Text characteristics
    if text.isupper():
        score += 1.0
    if text.endswith(':'):
        score += 1.0
    if not text.endswith('.'):
        score += 0.5
    
    return score

def enhance_ranked_sections(ranked_sections: List[Dict], section_map: Dict) -> List[Dict]:
    """Enhance ranked sections with proper text content and metadata."""
    enhanced_sections = []
    
    for section in ranked_sections:
        document = section['document']
        section_title = section['section_title']
        
        # Find the corresponding content from section_map
        section_text = ""
        section_level = "H1"
        
        if document in section_map:
            for original_section in section_map[document]:
                if original_section['title'] == section_title:
                    section_text = original_section['content']
                    section_level = original_section.get('level', 'H1')
                    break
        
        # Fallback: use section title as content if no content found
        if not section_text:
            section_text = section_title
        
        enhanced_section = section.copy()
        enhanced_section['text'] = section_text
        enhanced_section['level'] = section_level
        enhanced_section['content_length'] = len(section_text)
        
        enhanced_sections.append(enhanced_section)
    
    return enhanced_sections

def create_metadata(documents: List[Dict], structured_docs: List[Dict]) -> Dict:
    """Create comprehensive metadata for the final output - FIXED VERSION."""
    total_sections = sum(doc.get('document_stats', {}).get('total_sections', 0) 
                        for doc in structured_docs)
    total_pages = sum(doc.get('document_stats', {}).get('pages', 1) 
                     for doc in structured_docs)
    
    # FIX: Use actual filenames instead of titles
    input_documents = [doc["filename"] for doc in documents]
    
    metadata = {
        "input_documents": input_documents,  # FIXED: Now uses filenames
        "persona": {
            "role": "Travel Planner"  # This should be extracted from input
        },
        "job_to_be_done": {
            "task": "Plan a trip of 4 days for a group of 10 college friends."  # This should be extracted from input
        },
        "processing_timestamp": "",  # This should be set when called
        "processing_info": {
            "total_documents": len(documents),
            "documents_processed": len(structured_docs),
            "total_sections_extracted": total_sections,
            "total_pages_processed": total_pages,
            "processing_method": "enhanced_structure_extraction"
        },
        "document_details": []
    }
    
    # Add per-document details
    for doc in structured_docs:
        stats = doc.get('document_stats', {})
        metadata["document_details"].append({
            "filename": doc["filename"],
            "sections_found": stats.get('total_sections', 0),
            "headings_found": stats.get('total_headings', 0),
            "pages": stats.get('pages', 1),
            "text_blocks": stats.get('total_blocks', 0)
        })
    
    return metadata

def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Load input configuration
    try:
        input_data = load_input()
       
    except Exception as e:
        print(f"[ERROR] Failed to load input: {str(e)}")
        return
    
    persona = input_data["persona"]
    job = input_data["job_to_be_done"]
    documents = input_data["documents"]
    
    # Process documents
    structured_docs = process_documents(documents)
    
    if not structured_docs:
        print("[ERROR] No documents could be processed successfully")
        return
    
    # Create section map with enhanced filtering
    section_map = create_comprehensive_section_map(structured_docs)
    
    total_sections = sum(len(sections) for sections in section_map.values())
    
    
    # Rank sections using enhanced algorithm
    query = f"Persona: {persona}. Job to be done: {job}"
    ranked_sections = compute_similarity_scores(query, section_map)
    
    print(f"[INFO] Ranked {len(ranked_sections)} sections by relevance")
    
    # Enhance ranked sections with proper content
    enhanced_sections = enhance_ranked_sections(ranked_sections, section_map)

    # Take top sections for final output
    top_sections = enhanced_sections[:10]  # Top 10 most relevant sections
    
    # Generate summaries
    summaries = summarize_sections(top_sections)
    
    # Create metadata with proper document names
    metadata = create_metadata(documents, structured_docs)
    
    # Build final output
    final_output = build_final_output(
        persona=persona,
        job_to_be_done=job,
        extracted_sections=top_sections,
        summarized_sections=summaries,
        metadata=metadata
    )
    
    # Write output
    try:
        with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Output written to {OUTPUT_PATH}")
        
        # Debug output
        print(f"[DEBUG] Top 5 sections selected:")
        for i, section in enumerate(top_sections[:5], 1):
            print(f"  {i}. {section['section_title']} (from {section['document']})")
        
    except Exception as e:
        print(f"[ERROR] Failed to write output: {str(e)}")

if __name__ == "__main__":
    main()