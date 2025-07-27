# summarizer.py

import re
from typing import List, Dict
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

class FastTextRankSummarizer:
    """
    Extractive summarizer using TextRank (Sumy library).
    Lightweight and CPU-efficient. Works well for structured/unstructured docs.
    """
    def __init__(self, sentence_count: int = 3):
        self.sentence_count = sentence_count
        self.summarizer = TextRankSummarizer()

    def summarize_text(self, text: str) -> str:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary_sentences = self.summarizer(parser.document, self.sentence_count)
        return " ".join(str(sentence) for sentence in summary_sentences)


def clean_text(text: str) -> str:
    """
    Clean extracted section text: remove excessive whitespace, line breaks, etc.
    """
    return re.sub(r'\s+', ' ', text).strip()


def summarize_sections(extracted_sections: List[Dict]) -> List[Dict]:
    """
    Given ranked sections with content, return summarized version per section.
    
    Input format:
    [
        {
            'document': 'doc1.pdf',
            'section_title': 'Some Title',
            'text': 'Long content...',
            'page_number': 2
        },
        ...
    ]
    
    Output format:
    [
        {
            'document': 'doc1.pdf',
            'refined_text': 'Summary content...',
            'page_number': 2
        },
        ...
    ]
    """
    summarizer = FastTextRankSummarizer(sentence_count=3)
    summaries = []

    for section in extracted_sections:
        raw_text = clean_text(section.get("text", ""))
        summary = summarizer.summarize_text(raw_text)

        summaries.append({
            "document": section["document"],
            "refined_text": summary,
            "page_number": section["page_number"]
        })

    return summaries