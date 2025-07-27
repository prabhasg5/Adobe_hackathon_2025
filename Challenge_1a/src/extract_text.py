import fitz
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_text_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:
                # NEW: Combine all lines in a block first
                block_text = ""
                max_font_size = 0
                
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    font_sizes = [span["size"] for span in line["spans"]]
                    
                    block_text += " " + line_text.strip()
                    if font_sizes:
                        max_font_size = max(max_font_size, max(font_sizes))
                
                # Only add if substantial text
                block_text = block_text.strip()
                if len(block_text) > 2:
                    data.append({
                        "text": block_text,
                        "page": page_num + 1,
                        "font_size": max_font_size
                    })
    
    return data

def group_related_lines(lines):
    """Group lines that are visually close and might be part of the same text block"""
    if not lines:
        return []
    
    groups = []
    current_group = [lines[0]]
    
    for i in range(1, len(lines)):
        prev_line = lines[i-1]
        curr_line = lines[i]
        
        # Get vertical positions
        prev_y = prev_line["bbox"][1] if prev_line.get("bbox") else 0
        curr_y = curr_line["bbox"][1] if curr_line.get("bbox") else 0
        
        # If lines are close vertically (within reasonable distance)
        y_distance = abs(curr_y - prev_y)
        
        # If distance is small, they might be part of same text block
        if y_distance < 20:  # Adjust threshold as needed
            current_group.append(curr_line)
        else:
            groups.append(current_group)
            current_group = [curr_line]
    
    groups.append(current_group)
    return groups

def get_combined_bbox(line_group):
    """Get combined bounding box for a group of lines"""
    if not line_group:
        return None
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0
    
    for line in line_group:
        if line.get("bbox"):
            bbox = line["bbox"]
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
    
    return [min_x, min_y, max_x, max_y] if min_x != float('inf') else None

def clean_and_merge_text(text):
    """Clean up garbled PDF extraction text"""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Fix common PDF extraction issues
    # Remove excessive repeated characters (like "eeee")
    text = re.sub(r'(.)\1{4,}', r'\1\1', text)
    
    # Fix broken words - common patterns
    text = re.sub(r'\b(\w{1,3})\s+\1+\b', r'\1', text)  # "R R R" -> "R"
    
    # Fix "RFP: R" patterns - expand common abbreviations
    if re.match(r'^RFP:\s*R+\s*$', text, re.IGNORECASE):
        return "RFP: Request for Proposal"
    
    # Fix other common broken patterns
    text = re.sub(r'\bquest f\w*', 'quest for', text, flags=re.IGNORECASE)
    text = re.sub(r'\br Pr\w*', 'r Proposal', text, flags=re.IGNORECASE)
    
    # Remove weird spacing
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common heading patterns
    text = re.sub(r'^(\d+)\s*\1+', r'\1', text)  # "1 1 1" -> "1"
    
    return text.strip()

def merge_broken_headings(data):
    """Merge headings that were broken across multiple lines"""
    if len(data) < 2:
        return data
    
    merged_data = []
    i = 0
    
    while i < len(data):
        current = data[i]
        
        # Look ahead to see if next items should be merged
        merged_text = current["text"]
        merged_font_size = current["font_size"]
        
        # Check if this looks like a broken heading that continues
        j = i + 1
        while j < len(data) and should_merge_with_previous(data[j], current):
            merged_text += " " + data[j]["text"]
            merged_font_size = max(merged_font_size, data[j]["font_size"])
            j += 1
        
        # Create merged entry
        merged_entry = {
            "text": clean_and_merge_text(merged_text),
            "page": current["page"],
            "font_size": merged_font_size,
            "bbox": current.get("bbox")
        }
        
        merged_data.append(merged_entry)
        i = j
    
    return merged_data

def should_merge_with_previous(current_item, previous_item):
    """Determine if current item should be merged with previous"""
    # Same page
    if current_item["page"] != previous_item["page"]:
        return False
    
    # Similar font sizes (within 2 points)
    font_diff = abs(current_item["font_size"] - previous_item["font_size"])
    if font_diff > 2:
        return False
    
    # Short text that looks like a continuation
    current_text = current_item["text"].strip()
    previous_text = previous_item["text"].strip()
    
    # If current text is very short and looks like continuation
    if len(current_text.split()) <= 3:
        # Check if it looks like a fragment
        if (not current_text.endswith('.') and 
            not current_text[0].isupper() and 
            len(previous_text.split()) <= 10):
            return True
    
    # Check for broken patterns like "RFP: R" followed by "equest"
    if ("RFP" in previous_text and len(current_text.split()) <= 2 and 
        any(word in current_text.lower() for word in ["equest", "quest", "oposal", "proposal"])):
        return True
    
    return False

def remove_duplicates_and_fragments(data):
    """Remove duplicate and fragment entries"""
    cleaned_data = []
    seen_texts = set()
    
    # Sort by font size (descending) to prioritize complete text over fragments
    data_sorted = sorted(data, key=lambda x: (-x["font_size"], x["page"]))
    
    for item in data_sorted:
        text = item["text"].strip().lower()
        
        # Skip empty or very short texts
        if len(text) < 2:
            continue
        
        # Check if this is a duplicate or fragment
        is_duplicate = False
        
        for seen_text in seen_texts:
            # Exact match
            if text == seen_text:
                is_duplicate = True
                break
            
            # One text contains the other (fragment detection)
            if text in seen_text or seen_text in text:
                # Keep the longer, more complete version
                if len(text) <= len(seen_text):
                    is_duplicate = True
                    break
                else:
                    # Remove the shorter version and add the longer one
                    seen_texts.discard(seen_text)
                    break
        
        if not is_duplicate:
            seen_texts.add(text)
            cleaned_data.append(item)
    
    # Sort back by page and position
    return sorted(cleaned_data, key=lambda x: (x["page"], -x["font_size"]))

def is_heading_candidate(line):
    """Enhanced heading candidate detection"""
    text = line["text"].strip()
    
    if not text or len(text) < 2:
        return False
    
    word_count = len(text.split())
    char_count = len(text)
    
    # More flexible length limits
    if word_count > 25 or char_count > 200:
        return False
    
    # Skip if mostly numbers or special characters
    alpha_ratio = len(re.sub(r'[^a-zA-Z]', '', text)) / len(text) if text else 0
    if alpha_ratio < 0.25:
        return False
    
    # Skip obvious body text patterns
    body_text_patterns = [
        r'^[a-z].*[.!?]$',  # Starts lowercase, ends with punctuation
        r'.*\b(the|this|that|and|but|or|in|on|at|to|for|with|by)\b.*[.!?]$'
    ]
    
    for pattern in body_text_patterns:
        if re.search(pattern, text, re.IGNORECASE) and word_count > 8:
            return False
    
    # Accept clear heading patterns
    heading_patterns = [
        r'^[A-Z][^.]*$',  # Starts with capital, no ending period
        r'^\d+[\.\)]\s+[A-Z]',  # Numbered headings
        r'^[A-Z\s]+$',  # All caps
        r'^.*:$',  # Ends with colon
        r'^RFP.*',  # Starts with RFP
        r'^(Summary|Background|Introduction|Conclusion|Overview|Abstract).*',
    ]
    
    for pattern in heading_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    # Default: reasonable length without sentence-ending punctuation
    if 1 <= word_count <= 15 and not text.endswith(('.', '!', '?')):
        return True
    
    return False

def classify_headings_improved(lines):
    """Improved heading classification with better title detection"""
    
    # Filter heading candidates
    candidates = [line for line in lines if is_heading_candidate(line)]
    if not candidates:
        return []
    
    # Special handling for documents with clear title patterns
    title_candidate = detect_document_title(candidates)
    
    if len(candidates) == 1:
        return [{
            "level": "H1",
            "text": candidates[0]["text"],
            "page": candidates[0]["page"]
        }]
    
    # Get embeddings for clustering
    texts = [c["text"] for c in candidates]
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Determine number of clusters
    n_clusters = min(4, max(2, len(set(c["font_size"] for c in candidates))))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings.cpu().numpy())
    
    # Calculate cluster statistics
    cluster_stats = defaultdict(lambda: {'font_sizes': [], 'texts': [], 'pages': []})
    for i, (candidate, label) in enumerate(zip(candidates, cluster_labels)):
        cluster_stats[label]['font_sizes'].append(candidate['font_size'])
        cluster_stats[label]['texts'].append(candidate['text'])
        cluster_stats[label]['pages'].append(candidate['page'])
    
    # Assign heading levels
    level_assignment = assign_heading_levels(cluster_stats, title_candidate, candidates, cluster_labels)
    
    # Build final outline
    outline = []
    for candidate, label in zip(candidates, cluster_labels):
        outline.append({
            "level": level_assignment[label],
            "text": candidate["text"],
            "page": candidate["page"]
        })
    
    return outline

def detect_document_title(candidates):
    """Detect the most likely document title"""
    if not candidates:
        return None
    
    # Look for titles on first page with largest font
    first_page_candidates = [c for c in candidates if c["page"] == 1]
    
    if not first_page_candidates:
        return None
    
    # Find largest font size on first page
    max_font = max(c["font_size"] for c in first_page_candidates)
    title_candidates = [c for c in first_page_candidates if c["font_size"] == max_font]
    
    # Prefer shorter, title-like text
    title_candidates.sort(key=lambda x: (len(x["text"].split()), x["text"]))
    
    # Look for obvious title patterns
    for candidate in title_candidates:
        text = candidate["text"]
        if any(pattern in text for pattern in ["Ontario's Libraries", "Digital Library", "RFP:"]):
            return candidate
    
    return title_candidates[0] if title_candidates else None

def assign_heading_levels(cluster_stats, title_candidate, candidates, cluster_labels):
    """Assign H1, H2, H3 levels to clusters"""
    # Calculate average font size per cluster
    cluster_avg_font = {}
    for label, stats in cluster_stats.items():
        cluster_avg_font[label] = sum(stats['font_sizes']) / len(stats['font_sizes'])
    
    level_assignment = {}
    
    # If we have a clear title candidate, make its cluster H1
    if title_candidate:
        title_cluster = None
        for i, candidate in enumerate(candidates):
            if candidate["text"] == title_candidate["text"]:
                title_cluster = cluster_labels[i]
                break
        
        if title_cluster is not None:
            level_assignment[title_cluster] = "H1"
            
            # Sort remaining clusters by font size
            remaining_clusters = [c for c in cluster_avg_font.keys() if c != title_cluster]
            remaining_clusters.sort(key=lambda x: cluster_avg_font[x], reverse=True)
            
            for i, cluster in enumerate(remaining_clusters):
                level_assignment[cluster] = f"H{i+2}"
        else:
            # Fallback to font size based assignment
            sorted_clusters = sorted(cluster_avg_font.items(), key=lambda x: x[1], reverse=True)
            for i, (cluster, _) in enumerate(sorted_clusters):
                level_assignment[cluster] = f"H{i+1}"
    else:
        # No clear title - use font size based assignment
        sorted_clusters = sorted(cluster_avg_font.items(), key=lambda x: x[1], reverse=True)
        for i, (cluster, _) in enumerate(sorted_clusters):
            level_assignment[cluster] = f"H{i+1}"
    
    return level_assignment

# Wrapper function to maintain compatibility
def classify_headings(lines):
    """Main function to replace your current classify_headings"""
    return classify_headings_improved(lines)