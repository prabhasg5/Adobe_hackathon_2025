import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

def classify_headings(lines: List[Dict]) -> List[Dict]:
    if not lines:
        return []
    
    # Filter potential headings based on enhanced criteria
    heading_candidates = identify_heading_candidates(lines)
    
    if not heading_candidates:
        return []
    
    # Classify headings by hierarchy
    classified_headings = classify_heading_hierarchy(heading_candidates)
    
    # Sort by page and position
    classified_headings.sort(key=lambda x: (x["page"], x.get("position", 0)))
    
    return classified_headings

def identify_heading_candidates(lines: List[Dict]) -> List[Dict]:
    """
    Identify potential headings using multiple criteria.
    """
    candidates = []
    
    # Calculate document statistics for relative comparisons
    doc_stats = calculate_document_stats(lines)
    
    for line in lines:
        # Use heading_score if available (from enhanced extract_text)
        if "heading_score" in line:
            heading_likelihood = line["heading_score"]
        else:
            heading_likelihood = calculate_basic_heading_score(line, doc_stats)
        
        # Threshold for heading candidacy
        if heading_likelihood >= 2.0:
            candidate = line.copy()
            candidate["heading_likelihood"] = heading_likelihood
            candidates.append(candidate)
    
    return candidates

def calculate_document_stats(lines: List[Dict]) -> Dict:
    """Calculate document-wide statistics for comparison."""
    if not lines:
        return {}
    
    font_sizes = [line.get("font_size", 12) for line in lines]
    word_counts = [line.get("word_count", len(line.get("text", "").split())) for line in lines]
    
    return {
        "avg_font_size": sum(font_sizes) / len(font_sizes),
        "max_font_size": max(font_sizes),
        "min_font_size": min(font_sizes),
        "avg_word_count": sum(word_counts) / len(word_counts),
        "total_blocks": len(lines)
    }

def calculate_basic_heading_score(line: Dict, doc_stats: Dict) -> float:
    """
    Calculate heading likelihood score for lines without pre-calculated scores.
    """
    text = line.get("text", "")
    font_size = line.get("font_size", 12)
    page = line.get("page", 1)
    word_count = len(text.split())
    
    score = 0.0
    
    # Font size relative to document
    if doc_stats.get("max_font_size", 12) > 0:
        font_ratio = font_size / doc_stats["max_font_size"]
        if font_ratio >= 0.95:  # Near maximum font
            score += 3.0
        elif font_ratio >= 0.8:  # Large font
            score += 2.0
        elif font_ratio >= 0.6:  # Medium-large font
            score += 1.0
    
    # Text characteristics
    text_lower = text.lower().strip()
    
    # Length characteristics (headings are usually 1-15 words)
    if 1 <= word_count <= 8:
        score += 1.5
    elif 9 <= word_count <= 15:
        score += 1.0
    elif word_count > 20:
        score -= 1.0
    
    # Capitalization patterns
    if text.isupper():
        score += 1.0
    elif is_title_case_basic(text):
        score += 0.8
    
    # Structural indicators
    if text.endswith(':'):
        score += 1.2
    
    if re.match(r'^\d+[\.\)]\s+', text):  # Numbered sections
        score += 1.5
    
    if re.match(r'^[A-Z][^.!?]*$', text):  # Starts caps, no ending punctuation
        score += 0.8
    
    # Semantic indicators
    heading_keywords = [
        'introduction', 'overview', 'summary', 'conclusion', 'background',
        'methodology', 'results', 'discussion', 'chapter', 'section',
        'abstract', 'purpose', 'objective', 'scope', 'requirements',
        'specifications', 'implementation', 'analysis', 'recommendations'
    ]
    
    for keyword in heading_keywords:
        if keyword in text_lower:
            score += 1.0
            break
    
    # Position factors
    if page == 1:  # First page more likely to have titles
        score += 0.5
    
    # Penalty for obvious non-headings
    if text.endswith('.') and word_count > 10:
        score -= 2.0
    
    if any(char in text for char in ['@', 'http', 'www.']):
        score -= 3.0
    
    # Skip obvious metadata
    metadata_patterns = [
        r'^\d{1,2}/\d{1,2}/\d{4}',  # Dates
        r'^page\s+\d+',  # Page numbers
        r'^version\s+\d',  # Version numbers
        r'copyright|©'  # Copyright
    ]
    
    for pattern in metadata_patterns:
        if re.search(pattern, text_lower):
            score -= 2.0
            break
    
    return max(0.0, score)

def is_title_case_basic(text: str) -> bool:
    """Basic title case detection."""
    words = text.split()
    if len(words) < 2:
        return False
    
    capitalized = sum(1 for word in words if word and word[0].isupper())
    return capitalized >= len(words) * 0.7

def classify_heading_hierarchy(candidates: List[Dict]) -> List[Dict]:
    """
    Classify heading candidates into H1, H2, H3 hierarchy.
    """
    if not candidates:
        return []
    
    # Sort by heading likelihood (highest first)
    candidates.sort(key=lambda x: x.get("heading_likelihood", 0), reverse=True)
    
    # Group by font size and other characteristics
    font_groups = group_by_font_characteristics(candidates)
    
    # Assign heading levels
    heading_levels = assign_heading_levels(font_groups, candidates)
    
    # Create final heading list
    classified_headings = []
    for candidate in candidates:
        level = determine_heading_level(candidate, heading_levels)
        
        classified_headings.append({
            "level": level,
            "text": candidate["text"],
            "page": candidate["page"],
            "font_size": candidate.get("font_size", 12),
            "position": candidate.get("y", 0),
            "likelihood": candidate.get("heading_likelihood", 0)
        })
    
    return classified_headings

def group_by_font_characteristics(candidates: List[Dict]) -> Dict:
    """Group candidates by font size and formatting characteristics."""
    groups = defaultdict(list)
    
    for candidate in candidates:
        # Create a key based on font size (rounded to nearest 0.5)
        font_size = candidate.get("font_size", 12)
        font_key = round(font_size * 2) / 2
        
        # Consider bold/formatting if available
        is_bold = candidate.get("is_bold", False)
        formatting_key = f"{font_key}{'_bold' if is_bold else ''}"
        
        groups[formatting_key].append(candidate)
    
    return groups

def assign_heading_levels(font_groups: Dict, candidates: List[Dict]) -> Dict:
    """
    Assign H1, H2, H3 levels to font groups.
    """
    # Sort groups by font size (descending)
    sorted_groups = sorted(font_groups.items(), 
                          key=lambda x: extract_font_size_from_key(x[0]), 
                          reverse=True)
    
    level_assignment = {}
    
    # Assign levels (max 3 levels: H1, H2, H3)
    for i, (group_key, group_candidates) in enumerate(sorted_groups[:3]):
        level = f"H{i + 1}"
        level_assignment[group_key] = level
    
    # Any remaining groups get H3
    for group_key, _ in sorted_groups[3:]:
        level_assignment[group_key] = "H3"
    
    return level_assignment

def extract_font_size_from_key(key: str) -> float:
    """Extract font size from formatting key."""
    try:
        return float(key.split('_')[0])
    except (ValueError, IndexError):
        return 12.0

def determine_heading_level(candidate: Dict, level_assignment: Dict) -> str:
    """Determine the heading level for a specific candidate."""
    font_size = candidate.get("font_size", 12)
    is_bold = candidate.get("is_bold", False)
    
    # Create key matching the grouping logic
    font_key = round(font_size * 2) / 2
    formatting_key = f"{font_key}{'_bold' if is_bold else ''}"
    
    # Try exact match first
    if formatting_key in level_assignment:
        return level_assignment[formatting_key]
    
    # Try without bold formatting
    simple_key = str(font_key)
    if simple_key in level_assignment:
        return level_assignment[simple_key]
    
    # Fallback: determine by likelihood and font size
    likelihood = candidate.get("heading_likelihood", 0)
    
    if likelihood >= 4.0:
        return "H1"
    elif likelihood >= 3.0:
        return "H2"
    else:
        return "H3"

def extract_section_content(lines: List[Dict], headings: List[Dict]) -> List[Dict]:
    """
    Extract content that belongs to each heading section.
    """
    sections = []
    
    for i, heading in enumerate(headings):
        # Find content between this heading and the next
        current_page = heading["page"]
        current_position = heading.get("position", 0)
        
        # Determine end boundary
        if i < len(headings) - 1:
            next_heading = headings[i + 1]
            end_page = next_heading["page"]
            end_position = next_heading.get("position", float('inf'))
        else:
            end_page = float('inf')
            end_position = float('inf')
        
        # Collect content for this section
        section_content = []
        for line in lines:
            line_page = line.get("page", 1)
            line_position = line.get("y", 0)
            
            # Skip the heading itself
            if line.get("text", "") == heading["text"]:
                continue
            
            # Check if line belongs to this section
            if (line_page == current_page and line_position > current_position) or \
               (line_page > current_page and (line_page < end_page or 
                (line_page == end_page and line_position < end_position))):
                
                # Filter out obvious non-content (other headings, etc.)
                if not is_likely_heading_text(line.get("text", "")):
                    section_content.append(line.get("text", ""))
        
        # Combine content
        content_text = " ".join(section_content).strip()
        
        sections.append({
            "title": heading["text"],
            "content": content_text,
            "page_number": heading["page"],
            "level": heading["level"]
        })
    
    return sections

def is_likely_heading_text(text: str) -> bool:
    """Quick check if text is likely a heading (to avoid including in content)."""
    if not text or len(text.split()) > 15:
        return False
    
    # Check for heading-like characteristics
    if text.isupper() or text.endswith(':'):
        return True
    
    if re.match(r'^\d+[\.\)]\s+[A-Z]', text):
        return True
    
    return False

# Main function to maintain compatibility with existing code
def classify_headings_with_content(lines: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify headings and extract section content.
    
    Returns:
        Tuple of (headings, sections) where sections include content
    """
    headings = classify_headings(lines)
    sections = extract_section_content(lines, headings)
    
    return headings, sections