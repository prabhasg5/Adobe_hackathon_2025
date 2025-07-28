import json
from datetime import datetime
from typing import List, Dict, Any

def build_final_output(persona: str, job_to_be_done: str, 
                      extracted_sections: List[Dict], 
                      summarized_sections: List[Dict],
                      metadata: Dict) -> Dict:

    current_timestamp = datetime.now().isoformat()
    formatted_extracted_sections = []
    for i, section in enumerate(extracted_sections, 1):
        formatted_section = {
            "document": section.get('document', ''),
            "section_title": section.get('section_title', ''),
            "importance_rank": i,
            "page_number": section.get('page_number', 1)
        }
        formatted_extracted_sections.append(formatted_section)
    
    formatted_subsection_analysis = []
    for summary in summarized_sections:
        formatted_summary = {
            "document": summary.get('document', ''),
            "refined_text": summary.get('refined_text', ''),
            "page_number": summary.get('page_number', 1)
        }
        formatted_subsection_analysis.append(formatted_summary)

    final_output = {
        "metadata": {
            "input_documents": metadata.get("input_documents", []),
            "persona": {
                "role": persona
            },
            "job_to_be_done": {
                "task": job_to_be_done
            },
            "processing_timestamp": current_timestamp
        },
        "extracted_sections": formatted_extracted_sections,
        "subsection_analysis": formatted_subsection_analysis
    }
    
    return final_output

def validate_output_structure(output: Dict) -> bool:
    required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
    
    for key in required_keys:
        if key not in output:
            print(f"[ERROR] Missing required key: {key}")
            return False
    metadata = output["metadata"]
    required_metadata_keys = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
    
    for key in required_metadata_keys:
        if key not in metadata:
            print(f"[ERROR] Missing required metadata key: {key}")
            return False
    
    if not metadata["input_documents"]:
        print("[ERROR] input_documents should not be empty")
        return False

    if not output["extracted_sections"]:
        print("[ERROR] extracted_sections should not be empty")
        return False
    
    for i, section in enumerate(output["extracted_sections"]):
        required_section_keys = ["document", "section_title", "importance_rank", "page_number"]
        for key in required_section_keys:
            if key not in section:
                print(f"[ERROR] Missing key '{key}' in extracted_sections[{i}]")
                return False
    
    if not output["subsection_analysis"]:
        print("[ERROR] subsection_analysis should not be empty")
        return False
    
    for i, analysis in enumerate(output["subsection_analysis"]):
        required_analysis_keys = ["document", "refined_text", "page_number"]
        for key in required_analysis_keys:
            if key not in analysis:
                print(f"[ERROR] Missing key '{key}' in subsection_analysis[{i}]")
                return False
    
    print("SUCCESS")
    return True

def save_output_with_validation(output: Dict, filepath: str) -> bool:

    if not validate_output_structure(output):
        return False
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS]saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save output: {str(e)}")
        return False

