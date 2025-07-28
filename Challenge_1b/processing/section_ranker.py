import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter

# Choose a fast + accurate model for better semantic understanding
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def compute_similarity_scores(query: str, section_map: dict) -> list:
    """
    Generic ranking of sections by similarity to (persona + task) with adaptive scoring.
    
    Input:
        - query: concatenated persona + job description string
        - section_map: dict like {filename: [{"title": str, "content": str, "page_number": int, "level": str}]}
    
    Output:
        - ranked_sections: list of dicts sorted by score [{document, section_title, page_number, score, level}]
    """
    if not section_map:
        return []
    
    # Parse query to extract key requirements
    query_keywords = extract_key_terms(query)
    
    # Analyze document collection to understand content themes
    content_themes = analyze_document_themes(section_map)
    
    # Encode query once
    query_emb = model.encode(query, convert_to_tensor=True)
    
    results = []
    
    for doc_name, sections in section_map.items():
        for sec in sections:
            # Create enhanced text representation
            enhanced_text = create_enhanced_section_text(sec)
            
            # Compute semantic similarity
            sec_emb = model.encode(enhanced_text, convert_to_tensor=True)
            semantic_score = util.pytorch_cos_sim(query_emb, sec_emb).item()
            
            # Compute keyword-based relevance
            keyword_score = compute_keyword_relevance(sec, query, query_keywords)
            
            # Compute structural importance
            structural_score = compute_structural_importance(sec, sections)
            
            # Compute dynamic content relevance based on themes
            content_relevance_score = compute_dynamic_content_relevance(
                sec, query, content_themes
            )
            
            # Combine scores with adaptive weights
            final_score = combine_scores_adaptive(
                semantic_score, keyword_score, structural_score, 
                content_relevance_score, query
            )
            
            results.append({
                "document": doc_name,
                "section_title": sec['title'],
                "page_number": sec['page_number'],
                "score": final_score,
                "level": sec.get('level', 'H1'),
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "structural_score": structural_score,
                "content_relevance_score": content_relevance_score,
                "content_length": len(sec.get('content', ''))
            })
    
    # Sort by combined score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply adaptive diversity filtering
    results = apply_adaptive_diversity_filtering(results, query)
    
    return results

def analyze_document_themes(section_map: dict) -> Dict[str, float]:
    """
    Dynamically analyze the document collection to identify main themes.
    Returns a dict of theme keywords with their importance weights.
    """
    all_text = []
    
    # Collect all titles and content
    for doc_name, sections in section_map.items():
        for sec in sections:
            title = sec.get('title', '')
            content = sec.get('content', '')
            # Weight titles more heavily in theme analysis
            all_text.append(f"{title} {title} {content}")
    
    if not all_text:
        return {}
    
    # Combine all text
    combined_text = ' '.join(all_text).lower()
    
    # Extract meaningful terms (excluding stop words)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', combined_text)
    meaningful_words = [
        word for word in words 
        if len(word) > 3 and word not in stop_words and word.isalpha()
    ]
    
    # Count word frequencies
    word_counts = Counter(meaningful_words)
    
    # Get top themes (most frequent meaningful words)
    total_words = sum(word_counts.values())
    themes = {}
    
    for word, count in word_counts.most_common(50):  # Top 50 themes
        # Calculate relative frequency as importance weight
        weight = count / total_words
        themes[word] = weight
    
    return themes

def extract_key_terms(query: str) -> List[str]:
    """Extract key terms from the query for keyword matching."""
    query_lower = query.lower()
    
    # Remove common stop words and extract meaningful terms
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'persona', 'job', 'done'
    }
    
    # Extract words and phrases
    words = re.findall(r'\b\w+\b', query_lower)
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Extract multi-word phrases (2-3 words)
    phrases = []
    for i in range(len(words) - 1):
        if words[i] not in stop_words and words[i+1] not in stop_words:
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)
    
    return keywords + phrases

def create_enhanced_section_text(section: Dict) -> str:
    """Create enhanced text representation for better semantic matching."""
    title = section.get('title', '')
    content = section.get('content', '')
    
    # Weight title more heavily by repeating it
    enhanced_text = f"{title} {title} {content}"
    
    # Clean and normalize
    enhanced_text = re.sub(r'\s+', ' ', enhanced_text).strip()
    
    # Limit length to avoid overwhelming the model
    if len(enhanced_text) > 500:
        enhanced_text = enhanced_text[:500] + "..."
    
    return enhanced_text

def compute_keyword_relevance(section: Dict, query: str, query_keywords: List[str]) -> float:
    """Compute keyword-based relevance score."""
    title = section.get('title', '').lower()
    content = section.get('content', '').lower()
    combined_text = f"{title} {content}"
    
    if not query_keywords or not combined_text:
        return 0.0
    
    # Count keyword matches
    title_matches = 0
    content_matches = 0
    
    for keyword in query_keywords:
        keyword_lower = keyword.lower()
        
        # Count matches in title (weighted higher)
        title_count = title.count(keyword_lower)
        title_matches += title_count
        
        # Count matches in content
        content_count = content.count(keyword_lower)
        content_matches += content_count
    
    # Calculate score with title weighted 3x more than content
    title_score = min(title_matches * 3, 10)  # Cap at 10
    content_score = min(content_matches, 5)   # Cap at 5
    
    total_score = (title_score + content_score) / len(query_keywords)
    
    # Normalize to 0-1 range
    return min(total_score / 5.0, 1.0)

def compute_dynamic_content_relevance(section: Dict, query: str, themes: Dict[str, float]) -> float:
    """
    Compute content relevance dynamically based on discovered themes and query context.
    """
    title = section.get('title', '').lower()
    content = section.get('content', '').lower()
    combined_text = f"{title} {content}"
    
    if not themes:
        return 0.0
    
    # Analyze query to understand what type of content is needed
    query_lower = query.lower()
    
    # Determine query intent patterns
    is_planning_query = any(term in query_lower for term in [
        'plan', 'trip', 'visit', 'itinerary', 'schedule', 'organize'
    ])
    
    is_group_query = any(term in query_lower for term in [
        'group', 'friends', 'team', 'people', 'together'
    ])
    
    is_overview_needed = any(term in query_lower for term in [
        'overview', 'guide', 'comprehensive', 'complete', 'summary'
    ])
    
    score = 0.0
    
    # Score based on theme alignment
    for theme, theme_weight in themes.items():
        if theme in combined_text:
            # Base theme score
            theme_score = theme_weight * 10  # Scale up theme weights
            
            # Boost comprehensive/overview content for planning queries
            if is_planning_query or is_overview_needed:
                if any(overview_term in title for overview_term in [
                    'guide', 'comprehensive', 'overview', 'complete', 'major', 'main'
                ]):
                    theme_score *= 2.0
                
                # Boost higher-level sections for planning
                level = section.get('level', 'H1')
                if level == 'H1':
                    theme_score *= 1.5
                elif level == 'H2':
                    theme_score *= 1.2
            
            # Penalize overly detailed/specific content for group planning
            if is_group_query:
                if any(detail_term in combined_text for detail_term in [
                    'individual', 'personal', 'private', 'solo', 'alone'
                ]):
                    theme_score *= 0.5
            
            score += theme_score
    
    # Boost sections that appear to be introductory/comprehensive
    if any(intro_term in title for intro_term in [
        'introduction', 'overview', 'guide', 'comprehensive', 'complete'
    ]):
        score += 0.5
    
    # Boost sections with substantial content (likely more comprehensive)
    content_length = len(section.get('content', ''))
    if content_length > 200:
        score += 0.3
    elif content_length > 500:
        score += 0.5
    
    # Normalize score
    max_possible_score = sum(themes.values()) * 10 + 1.0  # +1 for bonuses
    normalized_score = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
    
    return normalized_score

def compute_structural_importance(section: Dict, all_sections: List[Dict]) -> float:
    """Compute structural importance based on section hierarchy and position."""
    score = 0.0
    
    # Level importance (H1 > H2 > H3)
    level = section.get('level', 'H1')
    if level == 'H1':
        score += 0.9  # Higher weight for top-level sections
    elif level == 'H2':
        score += 0.6
    elif level == 'H3':
        score += 0.3
    
    # Page importance (earlier pages often contain overviews)
    page = section.get('page_number', 1)
    if page == 1:
        score += 0.4
    elif page <= 3:
        score += 0.2
    elif page <= 5:
        score += 0.1
    
    # Content length importance (substantial content is often more comprehensive)
    content_length = len(section.get('content', ''))
    if 100 <= content_length <= 1000:
        score += 0.4
    elif content_length > 1000:
        score += 0.3
    elif 50 <= content_length < 100:
        score += 0.2
    
    # Title descriptiveness
    title = section.get('title', '')
    if title:
        title_words = len(title.split())
        if 2 <= title_words <= 8:
            score += 0.3
        elif title_words > 8:  # Very long titles might be too specific
            score += 0.1
    
    return min(score, 1.0)

def combine_scores_adaptive(semantic_score: float, keyword_score: float, 
                          structural_score: float, content_relevance_score: float, 
                          query: str) -> float:
    """Adaptively combine scores based on query characteristics."""
    
    # Analyze query to determine optimal weights
    query_lower = query.lower()
    
    # Default weights
    semantic_weight = 0.35
    keyword_weight = 0.25
    structural_weight = 0.2
    content_relevance_weight = 0.2
    
    # Adjust weights based on query type
    if any(term in query_lower for term in ['plan', 'organize', 'schedule']):
        # For planning queries, prioritize structural and content relevance
        structural_weight = 0.3
        content_relevance_weight = 0.3
        semantic_weight = 0.25
        keyword_weight = 0.15
    
    if any(term in query_lower for term in ['overview', 'summary', 'guide']):
        # For overview queries, prioritize structural importance
        structural_weight = 0.4
        content_relevance_weight = 0.25
        semantic_weight = 0.2
        keyword_weight = 0.15
    
    if any(term in query_lower for term in ['specific', 'detailed', 'particular']):
        # For specific queries, prioritize semantic and keyword matching
        semantic_weight = 0.4
        keyword_weight = 0.35
        structural_weight = 0.15
        content_relevance_weight = 0.1
    
    combined = (
        semantic_score * semantic_weight +
        keyword_score * keyword_weight +
        structural_score * structural_weight +
        content_relevance_score * content_relevance_weight
    )
    
    return combined

def apply_adaptive_diversity_filtering(results: List[Dict], query: str, max_results: int = 15) -> List[Dict]:
    """Apply adaptive diversity filtering based on query needs."""
    if not results:
        return results
    
    filtered_results = []
    doc_counts = Counter()
    
    # Determine diversity strategy based on query
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['comprehensive', 'complete', 'overview']):
        # For comprehensive queries, allow more sections per document
        max_per_doc = 3
    elif any(term in query_lower for term in ['group', 'team', 'friends']):
        # For group queries, ensure diverse content types
        max_per_doc = 2
    else:
        max_per_doc = 2
    
    # First pass: ensure we get top sections from different documents
    for result in results:
        doc_name = result['document']
        
        if doc_counts[doc_name] < max_per_doc:
            filtered_results.append(result)
            doc_counts[doc_name] += 1
        
        if len(filtered_results) >= max_results:
            break
    
    # If we still have room and high-scoring sections, add them
    if len(filtered_results) < max_results:
        for result in results:
            if result not in filtered_results and result['score'] > 0.5:
                filtered_results.append(result)
                if len(filtered_results) >= max_results:
                    break
    
    return filtered_results

# Enhanced function for better compatibility
def compute_similarity_scores_enhanced(query: str, section_map: dict, debug: bool = False) -> list:
    """Enhanced version with optional debugging."""
    results = compute_similarity_scores(query, section_map)
    
    if debug and results:
        debug_section_scores(results)
    
    return results

# Backward compatibility
def rank_sections_by_relevance(query: str, sections: List[Dict]) -> List[Dict]:
    """Alternative interface for direct section ranking."""
    # Convert to section_map format
    section_map = {"document": sections}
    
    # Rank and return
    results = compute_similarity_scores(query, section_map)
    
    # Remove document field for compatibility
    for result in results:
        if 'document' in result:
            del result['document']
    
    return results