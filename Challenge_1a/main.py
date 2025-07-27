import os
from collections import Counter

from src.extract_text import extract_text_with_metadata
from src.language_detector import detect_languages
from src.heading_ranker import classify_headings
from src.utils import save_output_json

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

def process_all_pdfs():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))

            # Step 1: Extract text lines with metadata
            lines = extract_text_with_metadata(pdf_path)

            # Step 2: Classify headings
            outline, detected_title = classify_headings(lines)

            # Step 3: Detect languages for each line
            lang_map = detect_languages(lines)
            all_languages = sorted(set(lang for lang in lang_map.values() if lang != "UNKNOWN"))
            language_distribution = dict(Counter(lang_map.values()))

            # Step 4: Determine title text
            title_text = "Untitled"
            if detected_title:
                title_text = detected_title["text"]
            elif outline:
                title_text = outline[0]["text"]

            # Step 5: Build clean outline (without individual language tags)
            clean_outline = []
            for item in outline:
                clean_item = {
                    "level": item["level"],
                    "text": item["text"],
                    "page": item["page"]
                }
                clean_outline.append(clean_item)

            # Step 6: Build final output
            output = {
                "title": title_text,
                "document_languages": all_languages,
                "language_distribution": language_distribution,
                "outline": clean_outline
            }

            # Step 7: Save output
            save_output_json(output, output_path)
            print(f"✓ Completed {filename} - Languages: {', '.join(all_languages)}")

if __name__ == "__main__":
    process_all_pdfs()
