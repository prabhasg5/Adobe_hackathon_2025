# json_builder.py
from typing import List, Dict
from datetime import datetime

def build_final_output(
    persona: str,
    job_to_be_done: str,
    extracted_sections: List[Dict],
    summarized_sections: List[Dict],
    metadata: Dict
) -> Dict:
    """
    Construct the final output JSON structure matching the desired format.
    
    Parameters:
    - persona: description of the target user
    - job_to_be_done: what the user is trying to accomplish
    - extracted_sections: List of ranked sections with title, text, page_number
    - summarized_sections: List of summaries corresponding to those sections
    - metadata: Dictionary containing document-level metadata
    
    Returns:
    A dictionary with the required output fields.
    """
    
    # Build metadata section
    metadata_output = {
        "input_documents": metadata.get("document_titles", []),
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    # Build extracted_sections with importance_rank
    extracted_sections_output = []
    for i, section in enumerate(extracted_sections, 1):
        extracted_sections_output.append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": i,
            "page_number": section["page_number"]
        })
    
    # Build subsection_analysis from summarized sections
    subsection_analysis_output = []
    for summary in summarized_sections:
        subsection_analysis_output.append({
            "document": summary["document"],
            "refined_text": summary["refined_text"],
            "page_number": summary["page_number"]
        })
    
    # Final output structure - matches your desired format exactly
    output = {
        "metadata": metadata_output,
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }
    
    return output