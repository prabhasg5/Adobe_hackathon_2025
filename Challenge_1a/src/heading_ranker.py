import re
from collections import defaultdict

# Removed SentenceTransformer and KMeans imports since they're not used in current logic

def clean_text(text):
    """Simple text cleaning"""
    text = text.strip()
    # Remove excessive repeated characters
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    # Remove duplicate numbers like "2222"
    text = re.sub(r'\b(\d)\1{2,}\b', r'\1', text)
    return text

def is_heading_candidate(line):
    """Simplified heading detection with early returns"""
    text = line.get("text", "").strip()
    
    # Early returns for obvious non-candidates
    if not text or len(text) < 3:
        return False
    
    # Quick character count check before expensive operations
    if len(text) > 200:  # Very long text
        return False
    
    word_count = len(text.split())
    
    # Skip very long text
    if word_count > 20:
        return False
    
    # Quick alpha ratio check (avoid regex if possible)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.4:
        return False
    
    # Skip obvious body text (ends with period and long)
    if text.endswith('.') and word_count > 10:
        return False
    
    return True

def is_likely_title(text, page, font_size, max_font_size, position_y=None):
    """Improved title detection logic"""
    # Must be on first page
    if page != 1:
        return False
    
    # Must have largest or near-largest font (within 2 points)
    if font_size < max_font_size - 2:
        return False
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Skip obvious non-titles
    non_title_patterns = [
        r'^(page|march|april|version|date|time)[\s\d]',
        r'^\d+[\.\)]',  # numbered items
        r'^[a-z]+@',    # email
        r'^www\.',      # website
        r'^http',       # url
        r'^\(\d',       # phone number pattern
        r'^address:?$', # standalone address
        r'^rsvp:?',     # rsvp text
    ]
    
    for pattern in non_title_patterns:
        if re.match(pattern, text_lower):
            return False
    
    # Prefer titles that are:
    # 1. Not too short (unless very specific keywords)
    # 2. Not addresses/contact info
    # 3. More substantial content
    
    # Skip very short text unless it's clearly a document type
    if word_count <= 2:
        title_indicators = ['overview', 'introduction', 'summary', 'guide', 'manual', 'report']
        if not any(indicator in text_lower for indicator in title_indicators):
            return False
    
    # Skip address-like content
    if any(indicator in text_lower for indicator in ['address:', 'phone:', 'email:', 'website:']):
        return False
    
    # Prefer longer, more descriptive titles (3-15 words)
    if 3 <= word_count <= 15:
        return True
    
    # Single word titles only if they're meaningful
    if word_count == 1:
        meaningful_single_words = [
            'overview', 'introduction', 'summary', 'guide', 'manual', 
            'report', 'analysis', 'proposal', 'specification'
        ]
        return text_lower in meaningful_single_words
    
    return False

def is_date_or_metadata(text):
    """Check if text is date or metadata (should be lower priority)"""
    text_lower = text.lower()
    
    # Date patterns
    if re.match(r'.*\d{4}.*', text) and len(text.split()) <= 4:
        return True
    
    # Version numbers
    if re.match(r'^version\s+[\d\.]+', text_lower):
        return True
    
    # Page numbers, references
    if re.match(r'^(page|march|april|rfp:.*\d+).*', text_lower):
        return True
    
    # Contact info patterns
    contact_patterns = [
        r'^\(\d{3}\)',  # phone
        r'^www\.',      # website
        r'@.*\.com',    # email
        r'^address:',   # address label
        r'^phone:',     # phone label
        r'^email:',     # email label
    ]
    
    for pattern in contact_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def find_document_title(candidates):
    """Find the most likely document title"""
    first_page = [c for c in candidates if c["page"] == 1]
    if not first_page:
        return None
    
    max_font = max(c["font_size"] for c in first_page)
    
    # Look for title candidates with largest font
    title_candidates = []
    for candidate in first_page:
        if is_likely_title(candidate["text"], candidate["page"], 
                          candidate["font_size"], max_font):
            title_candidates.append(candidate)
    
    if title_candidates:
        # Scoring system for better title selection
        def score_title_candidate(candidate):
            text = candidate["text"]
            font_size = candidate["font_size"]
            word_count = len(text.split())
            
            score = 0
            
            # Font size score (larger = better)
            score += (font_size - max_font + 5) * 2
            
            # Position score (higher on page = more likely title)
            if "y" in candidate and candidate["y"] < 300:
                score += 3
            
            # Word count score (prefer 3-10 words)
            if 3 <= word_count <= 10:
                score += 10
            elif word_count == 2:
                score += 5
            elif word_count == 1:
                score += 3
            
            # Content quality score
            text_lower = text.lower()
            
            # Boost for title-like words
            title_words = [
                'overview', 'introduction', 'guide', 'manual', 'report', 
                'analysis', 'specification', 'proposal', 'testing', 
                'qualification', 'foundation', 'level', 'extension'
            ]
            
            for word in title_words:
                if word in text_lower:
                    score += 3
            
            # Penalize metadata-like content
            if is_date_or_metadata(text):
                score -= 10
            
            return score
        
        # Select highest scoring title
        title_candidates.sort(key=score_title_candidate, reverse=True)
        return title_candidates[0]
    
    # Fallback: use largest font item that's not clearly metadata
    non_metadata = [c for c in first_page if not is_date_or_metadata(c["text"])]
    if non_metadata:
        # Sort by font size, then by position (assuming earlier = more likely title)
        non_metadata.sort(key=lambda x: (-x["font_size"], x.get("y", 0)))
        return non_metadata[0]
    
    return None

def merge_title_fragments(candidates):
    """Merge broken title fragments back together"""
    if len(candidates) < 2:
        return candidates
    
    # Look for title fragments on page 1 with similar font sizes
    page1_candidates = [c for c in candidates if c["page"] == 1]
    if len(page1_candidates) < 2:
        return candidates
    
    merged_candidates = []
    i = 0
    
    while i < len(page1_candidates):
        current = page1_candidates[i]
        current_text = current["text"].strip()
        
        # Check if this looks like start of title
        title_starters = [
            "to present", "rfp", "request for", "developing", "international",
            "software testing", "qualification", "foundation level"
        ]
        is_title_start = any(starter in current_text.lower() for starter in title_starters)
        
        if is_title_start and i < len(page1_candidates) - 1:
            # Try to merge with next items if they have similar font size
            merged_text = current_text
            font_size = current["font_size"]
            j = i + 1
            
            # Merge consecutive items with similar font size
            while j < len(page1_candidates):
                next_item = page1_candidates[j]
                font_diff = abs(next_item["font_size"] - font_size)
                
                # If similar font size (within 1 point) and reasonable length
                if font_diff <= 1 and len(next_item["text"].split()) <= 10:
                    merged_text += " " + next_item["text"].strip()
                    j += 1
                else:
                    break
            
            # Create merged title
            merged_title = {
                "text": merged_text,
                "page": current["page"],
                "font_size": font_size
            }
            merged_candidates.append(merged_title)
            
            # Skip the items we merged
            i = j
        else:
            merged_candidates.append(current)
            i += 1
    
    # Add remaining candidates from other pages
    other_pages = [c for c in candidates if c["page"] != 1]
    merged_candidates.extend(other_pages)
    
    return merged_candidates

def classify_headings(lines):
    """Optimized heading classification function"""
    # Pre-filter candidates more aggressively
    candidates = []
    for line in lines:
        if is_heading_candidate(line):
            # Clean text only for actual candidates
            cleaned_text = clean_text(line["text"])
            if cleaned_text and len(cleaned_text) >= 3:  # Double-check after cleaning
                line_copy = line.copy()
                line_copy["text"] = cleaned_text
                candidates.append(line_copy)
    
    if not candidates:
        return [], None
    
    # Early exit for very simple documents
    if len(candidates) == 1:
        return [{
            "level": "H1",
            "text": candidates[0]["text"],
            "page": candidates[0]["page"]
        }], candidates[0]
    
    # Skip expensive merging for small candidate lists
    if len(candidates) <= 5:
        title = find_document_title(candidates)
    else:
        # Only do expensive merging for complex documents
        candidates = merge_title_fragments(candidates)
        title = find_document_title(candidates)
    
    # Rest of the function remains the same...
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
    
    # Simplified font clustering
    font_sizes = [c["font_size"] for c in candidates]
    unique_fonts = sorted(set(font_sizes), reverse=True)
    
    # Limit to 3 font levels max
    font_to_level = {}
    for i, font_size in enumerate(unique_fonts[:3]):
        font_to_level[font_size] = f"H{i+1}"
    
    # Assign levels
    outline = []
    for candidate in candidates:
        font_size = candidate["font_size"]
        
        # Find closest font level
        closest_font = min(font_to_level.keys(), 
                          key=lambda x: abs(x - font_size))
        level = font_to_level[closest_font]
        
        text = candidate["text"]
        
        # Quick adjustments
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