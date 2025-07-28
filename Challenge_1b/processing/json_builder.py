import json
from datetime import datetime
from typing import List, Dict, Any

def build_final_output(persona: str, job_to_be_done: str, 
                      extracted_sections: List[Dict], 
                      summarized_sections: List[Dict],
                      metadata: Dict) -> Dict:
    """
    Build the final JSON output matching the ground truth format.
    
    Args:
        persona: The persona string
        job_to_be_done: The job description string
        extracted_sections: List of top ranked sections
        summarized_sections: List of summarized sections
        metadata: Metadata dictionary
    
    Returns:
        Final JSON structure matching expected format
    """
    
    # Create timestamp
    current_timestamp = datetime.now().isoformat()
    
    # Build extracted_sections in the correct format
    formatted_extracted_sections = []
    for i, section in enumerate(extracted_sections, 1):
        formatted_section = {
            "document": section.get('document', ''),
            "section_title": section.get('section_title', ''),
            "importance_rank": i,
            "page_number": section.get('page_number', 1)
        }
        formatted_extracted_sections.append(formatted_section)
    
    # Build subsection_analysis in the correct format
    formatted_subsection_analysis = []
    for summary in summarized_sections:
        formatted_summary = {
            "document": summary.get('document', ''),
            "refined_text": summary.get('refined_text', ''),
            "page_number": summary.get('page_number', 1)
        }
        formatted_subsection_analysis.append(formatted_summary)
    
    # Build the final output structure
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
    """
    Validate that the output matches the expected structure.
    
    Args:
        output: The output dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
    
    # Check top-level keys
    for key in required_keys:
        if key not in output:
            print(f"[ERROR] Missing required key: {key}")
            return False
    
    # Check metadata structure
    metadata = output["metadata"]
    required_metadata_keys = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
    
    for key in required_metadata_keys:
        if key not in metadata:
            print(f"[ERROR] Missing required metadata key: {key}")
            return False
    
    # Check that input_documents is not empty
    if not metadata["input_documents"]:
        print("[ERROR] input_documents should not be empty")
        return False
    
    # Check extracted_sections structure
    if not output["extracted_sections"]:
        print("[ERROR] extracted_sections should not be empty")
        return False
    
    for i, section in enumerate(output["extracted_sections"]):
        required_section_keys = ["document", "section_title", "importance_rank", "page_number"]
        for key in required_section_keys:
            if key not in section:
                print(f"[ERROR] Missing key '{key}' in extracted_sections[{i}]")
                return False
    
    # Check subsection_analysis structure
    if not output["subsection_analysis"]:
        print("[ERROR] subsection_analysis should not be empty")
        return False
    
    for i, analysis in enumerate(output["subsection_analysis"]):
        required_analysis_keys = ["document", "refined_text", "page_number"]
        for key in required_analysis_keys:
            if key not in analysis:
                print(f"[ERROR] Missing key '{key}' in subsection_analysis[{i}]")
                return False
    
    print("[SUCCESS] Output structure validation passed")
    return True

def save_output_with_validation(output: Dict, filepath: str) -> bool:
    """
    Save output with validation.
    
    Args:
        output: The output dictionary
        filepath: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    # Validate structure first
    if not validate_output_structure(output):
        return False
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Output saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save output: {str(e)}")
        return False

def debug_output_structure(output: Dict) -> None:
    """
    Print debug information about the output structure.
    
    Args:
        output: The output dictionary to debug
    """
    print("\n[DEBUG] Output Structure Analysis:")
    print("-" * 50)
    
    # Metadata info
    metadata = output.get("metadata", {})
    print(f"Input Documents: {len(metadata.get('input_documents', []))}")
    for doc in metadata.get('input_documents', []):
        print(f"  - {doc}")
    
    print(f"Persona: {metadata.get('persona', {}).get('role', 'Not set')}")
    print(f"Job: {metadata.get('job_to_be_done', {}).get('task', 'Not set')}")
    
    # Extracted sections info
    extracted = output.get("extracted_sections", [])
    print(f"\nExtracted Sections: {len(extracted)}")
    for i, section in enumerate(extracted[:5], 1):  # Show first 5
        print(f"  {i}. {section.get('section_title', 'No title')[:60]}...")
        print(f"     Document: {section.get('document', 'Unknown')}")
        print(f"     Page: {section.get('page_number', 'Unknown')}")
    
    # Subsection analysis info
    analysis = output.get("subsection_analysis", [])
    print(f"\nSubsection Analysis: {len(analysis)}")
    for i, sub in enumerate(analysis[:3], 1):  # Show first 3
        refined_text = sub.get('refined_text', 'No text')
        preview = refined_text[:100] + "..." if len(refined_text) > 100 else refined_text
        print(f"  {i}. {preview}")
        print(f"     Document: {sub.get('document', 'Unknown')}")
    
    print("-" * 50)