import fitz
import re
from collections import defaultdict
from typing import List, Dict, Tuple

def extract_text_with_metadata(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    data = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:
                block_info = extract_block_info(block, page_num + 1)
                if block_info and is_valid_text_block(block_info):
                    data.append(block_info)
    
    doc.close()
    data = merge_text_fragments(data)
    data = remove_duplicates_and_noise(data)
    data = enhance_with_structure_info(data)
    
    return data

def extract_block_info(block: dict, page_num: int) -> Dict:
    block_text = ""
    font_sizes = []
    font_names = []
    is_bold = False
    is_italic = False
    
    bbox = block.get("bbox", [0, 0, 0, 0])
    for line in block.get("lines", []):
        line_text = ""
        for span in line.get("spans", []):
            span_text = span.get("text", "").strip()
            if span_text:
                line_text += span_text + " "
                font_sizes.append(span.get("size", 12))
                font_names.append(span.get("font", ""))
            
                font_flags = span.get("flags", 0)
                if font_flags & 2**4: 
                    is_bold = True
                if font_flags & 2**1:
                    is_italic = True
        
        if line_text.strip():
            block_text += line_text.strip() + " "
    block_text = clean_extracted_text(block_text.strip())
    
    if not block_text or len(block_text) < 2:
        return None
    
    return {
        "text": block_text,
        "page": page_num,
        "font_size": max(font_sizes) if font_sizes else 12,
        "avg_font_size": sum(font_sizes) / len(font_sizes) if font_sizes else 12,
        "font_name": most_common_font(font_names),
        "is_bold": is_bold,
        "is_italic": is_italic,
        "bbox": bbox,
        "x": bbox[0],
        "y": bbox[1],
        "width": bbox[2] - bbox[0],
        "height": bbox[3] - bbox[1],
        "char_count": len(block_text),
        "word_count": len(block_text.split())
    }

def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  
    text = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', text)
    text = re.sub(r'(.)\1{4,}', r'\1', text)
    text = re.sub(r'\b(\w{1,2})\s+(\w{1,2})\s+(\w{1,2})\b', 
                  lambda m: m.group(1) + m.group(2) + m.group(3) 
                  if len(m.group(1) + m.group(2) + m.group(3)) < 15 else m.group(0), text)
    
    return text.strip()

def most_common_font(font_names: List[str]) -> str:
    if not font_names:
        return ""
    return max(set(font_names), key=font_names.count)

def is_valid_text_block(block_info: Dict) -> bool:
    text = block_info["text"]
    
    if len(text) < 3:
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < len(text) * 0.3:
        return False
    if re.match(r'^\d+$', text.strip()): 
        return False
    
    if text.lower().strip() in ['page', 'header', 'footer']:
        return False
    
    return True

def merge_text_fragments(data: List[Dict]) -> List[Dict]:
    if len(data) < 2:
        return data
    
    merged_data = []
    i = 0
    data.sort(key=lambda x: (x["page"], x["y"]))
    
    while i < len(data):
        current = data[i]
        merged_text = current["text"]
        merged_item = current.copy()
        j = i + 1
        while j < len(data) and should_merge_blocks(current, data[j]):
            merged_text += " " + data[j]["text"]
            merged_item["text"] = merged_text
            merged_item["font_size"] = max(merged_item["font_size"], data[j]["font_size"])
            merged_item["word_count"] = len(merged_text.split())
            merged_item["char_count"] = len(merged_text)
            j += 1
        
        merged_data.append(merged_item)
        i = j
    
    return merged_data

def should_merge_blocks(block1: Dict, block2: Dict) -> bool:
    if block1["page"] != block2["page"]:
        return False
    y_distance = abs(block1["y"] - block2["y"])
    if y_distance > 30:
        return False

    font_size_diff = abs(block1["font_size"] - block2["font_size"])
    if font_size_diff > 2:
        return False
    
    if block1["word_count"] > 10 or block2["word_count"] > 10:
        return False
    
    combined_text = block1["text"] + " " + block2["text"]
    if len(combined_text.split()) > 20:  
        return False
    
    return True

def remove_duplicates_and_noise(data: List[Dict]) -> List[Dict]:
    seen_texts = set()
    cleaned_data = []
    
    for item in data:
        text_key = item["text"].lower().strip()
        
        if text_key in seen_texts:
            continue
    
        if is_noise_text(item["text"]):
            continue
        
        seen_texts.add(text_key)
        cleaned_data.append(item)
    
    return cleaned_data

def is_noise_text(text: str) -> bool:
    """Identify noise text that should be filtered out."""
    text_lower = text.lower().strip()
    
    if len(text_lower) < 3:
        return True
    
 
    if re.match(r'^[\d\s\-_.()]+$', text_lower):
        return True
    
    noise_patterns = [
        r'^page\s*\d*$',
        r'^\d+$',  
        r'^[ivx]+$',  
        r'^©.*copyright',
        r'^all rights reserved',
        r'^\s*$',  
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, text_lower):
            return True
    
    return False

def enhance_with_structure_info(data: List[Dict]) -> List[Dict]:
    if not data:
        return data

    all_font_sizes = [item["font_size"] for item in data]
    avg_font_size = sum(all_font_sizes) / len(all_font_sizes)
    max_font_size = max(all_font_sizes)
    
    for item in data:
        item["font_size_ratio"] = item["font_size"] / avg_font_size
        item["is_large_font"] = item["font_size"] > avg_font_size * 1.2
        item["is_max_font"] = item["font_size"] == max_font_size
        item["is_top_of_page"] = item["y"] < 150 
        item["is_left_aligned"] = item["x"] < 100  
        item["is_title_case"] = is_title_case(item["text"])
        item["is_all_caps"] = item["text"].isupper()
        item["ends_with_colon"] = item["text"].endswith(':')
        item["is_numbered"] = bool(re.match(r'^\d+[\.\)]', item["text"]))
        item["heading_score"] = calculate_heading_score(item)
    
    return data

def is_title_case(text: str) -> bool:
    words = text.split()
    if len(words) < 2:
        return False
    
    title_case_count = sum(1 for word in words if word[0].isupper())
    return title_case_count >= len(words) * 0.7

def calculate_heading_score(item: Dict) -> float:
    score = 0.0
    
    if item["is_max_font"]:
        score += 3.0
    elif item["is_large_font"]:
        score += 2.0
    if item["is_bold"]:
        score += 1.5
    if item["is_all_caps"]:
        score += 1.0
    if item["is_title_case"]:
        score += 0.5
    if item["ends_with_colon"]:
        score += 1.0
    if item["is_numbered"]:
        score += 1.0
    if item["is_top_of_page"]:
        score += 0.5
    if item["is_left_aligned"]:
        score += 0.3

    word_count = item["word_count"]
    if 2 <= word_count <= 8:
        score += 1.0
    elif word_count == 1:
        score += 0.5
    elif word_count > 15:
        score -= 1.0
    
    text_lower = item["text"].lower()
    heading_keywords = [
        'introduction', 'overview', 'summary', 'conclusion', 'background',
        'methodology', 'results', 'discussion', 'chapter', 'section'
    ]
    
    for keyword in heading_keywords:
        if keyword in text_lower:
            score += 0.8
            break
    
    return max(0.0, score)