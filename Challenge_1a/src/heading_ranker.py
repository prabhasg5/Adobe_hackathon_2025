import re
from collections import defaultdict
def clean_text(text):
    text = text.strip()
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    text = re.sub(r'\b(\d)\1{2,}\b', r'\1', text)
    return text

def is_heading_candidate(line):
    text = line.get("text", "").strip()

    if not text or len(text) < 3:
        return False

    if len(text) > 200:  
        return False
    
    word_count = len(text.split())

    if word_count > 20:
        return False

    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.4:
        return False
    
    if text.endswith('.') and word_count > 10:
        return False
    
    return True

def is_likely_title(text, page, font_size, max_font_size, position_y=None):
    if page != 1:
        return False
    if font_size < max_font_size - 2:
        return False
    
    text_lower = text.lower()
    word_count = len(text.split())
    non_title_patterns = [
        r'^(page|march|april|version|date|time)[\s\d]',
        r'^\d+[\.\)]', 
        r'^[a-z]+@',   
        r'^www\.',     
        r'^http',      
        r'^\(\d',       
        r'^address:?$',
        r'^rsvp:?', 
    ]
    
    for pattern in non_title_patterns:
        if re.match(pattern, text_lower):
            return False

    if word_count <= 2:
        title_indicators = ['overview', 'introduction', 'summary', 'guide', 'manual', 'report']
        if not any(indicator in text_lower for indicator in title_indicators):
            return False
    
    if any(indicator in text_lower for indicator in ['address:', 'phone:', 'email:', 'website:']):
        return False

    if 3 <= word_count <= 15:
        return True
    
    if word_count == 1:
        meaningful_single_words = [
            'overview', 'introduction', 'summary', 'guide', 'manual', 
            'report', 'analysis', 'proposal', 'specification'
        ]
        return text_lower in meaningful_single_words
    
    return False

def is_date_or_metadata(text):
    text_lower = text.lower()
    if re.match(r'.*\d{4}.*', text) and len(text.split()) <= 4:
        return True

    if re.match(r'^version\s+[\d\.]+', text_lower):
        return True

    if re.match(r'^(page|march|april|rfp:.*\d+).*', text_lower):
        return True
    
    contact_patterns = [
        r'^\(\d{3}\)', 
        r'^www\.',      
        r'@.*\.com',    
        r'^address:',  
        r'^phone:',    
        r'^email:',    
    ]
    
    for pattern in contact_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def find_document_title(candidates):
    first_page = [c for c in candidates if c["page"] == 1]
    if not first_page:
        return None
    
    max_font = max(c["font_size"] for c in first_page)
    
    title_candidates = []
    for candidate in first_page:
        if is_likely_title(candidate["text"], candidate["page"], 
                          candidate["font_size"], max_font):
            title_candidates.append(candidate)
    
    if title_candidates:
        def score_title_candidate(candidate):
            text = candidate["text"]
            font_size = candidate["font_size"]
            word_count = len(text.split())
            score = 0
            score += (font_size - max_font + 5) * 2
            if "y" in candidate and candidate["y"] < 300:
                score += 3

            if 3 <= word_count <= 10:
                score += 10
            elif word_count == 2:
                score += 5
            elif word_count == 1:
                score += 3
            
            text_lower = text.lower()
            title_words = [
                'overview', 'introduction', 'guide', 'manual', 'report', 
                'analysis', 'specification', 'proposal', 'testing', 
                'qualification', 'foundation', 'level', 'extension'
            ]
            
            for word in title_words:
                if word in text_lower:
                    score += 3
            
            if is_date_or_metadata(text):
                score -= 10
            
            return score
        
        title_candidates.sort(key=score_title_candidate, reverse=True)
        return title_candidates[0]

    non_metadata = [c for c in first_page if not is_date_or_metadata(c["text"])]
    if non_metadata:
        non_metadata.sort(key=lambda x: (-x["font_size"], x.get("y", 0)))
        return non_metadata[0]
    
    return None

def merge_title_fragments(candidates):
    if len(candidates) < 2:
        return candidates
    page1_candidates = [c for c in candidates if c["page"] == 1]
    if len(page1_candidates) < 2:
        return candidates
    
    merged_candidates = []
    i = 0
    
    while i < len(page1_candidates):
        current = page1_candidates[i]
        current_text = current["text"].strip()
        title_starters = [
            "to present", "rfp", "request for", "developing", "international",
            "software testing", "qualification", "foundation level"
        ]
        is_title_start = any(starter in current_text.lower() for starter in title_starters)
        
        if is_title_start and i < len(page1_candidates) - 1:
            merged_text = current_text
            font_size = current["font_size"]
            j = i + 1
            
            while j < len(page1_candidates):
                next_item = page1_candidates[j]
                font_diff = abs(next_item["font_size"] - font_size)
                
                if font_diff <= 1 and len(next_item["text"].split()) <= 10:
                    merged_text += " " + next_item["text"].strip()
                    j += 1
                else:
                    break
            
            merged_title = {
                "text": merged_text,
                "page": current["page"],
                "font_size": font_size
            }
            merged_candidates.append(merged_title)
            i = j
        else:
            merged_candidates.append(current)
            i += 1
    other_pages = [c for c in candidates if c["page"] != 1]
    merged_candidates.extend(other_pages)
    
    return merged_candidates

def classify_headings(lines):
    candidates = []
    for line in lines:
        if is_heading_candidate(line):
            cleaned_text = clean_text(line["text"])
            if cleaned_text and len(cleaned_text) >= 3:
                line_copy = line.copy()
                line_copy["text"] = cleaned_text
                candidates.append(line_copy)
    
    if not candidates:
        return [], None
    
    if len(candidates) == 1:
        return [{
            "level": "H1",
            "text": candidates[0]["text"],
            "page": candidates[0]["page"]
        }], candidates[0]
    
    if len(candidates) <= 5:
        title = find_document_title(candidates)
    else:
        candidates = merge_title_fragments(candidates)
        title = find_document_title(candidates)
    if len(candidates) <= 2:
        result = []
        for i, candidate in enumerate(candidates):
            if title and candidate["text"] == title["text"]:
                level = "H1"
            elif i == 0 and not is_date_or_metadata(candidate["text"]):
                level = "H1"
            else:
                level = "H2"
            
            result.append({
                "level": level,
                "text": candidate["text"],
                "page": candidate["page"]
            })
        return result, title
    
    font_sizes = [c["font_size"] for c in candidates]
    unique_fonts = sorted(set(font_sizes), reverse=True)
    font_to_level = {}
    for i, font_size in enumerate(unique_fonts[:3]):
        font_to_level[font_size] = f"H{i+1}"
    
    outline = []
    for candidate in candidates:
        font_size = candidate["font_size"]
        closest_font = min(font_to_level.keys(), 
                          key=lambda x: abs(x - font_size))
        level = font_to_level[closest_font]
        
        text = candidate["text"]
        if is_date_or_metadata(text):
            level = "H3"
        
        if title and candidate["text"] == title["text"]:
            level = "H1"
        
        outline.append({
            "level": level,
            "text": text,
            "page": candidate["page"]
        })
    
    return outline, title