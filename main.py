import os
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

            # Step 1: Extract all lines with font size & page
            lines = extract_text_with_metadata(pdf_path)

            # Step 2: Detect languages for multilingual support
            lang_map = detect_languages(lines)

            # Step 3: Classify headings using sentence transformer + clustering
            outline = classify_headings(lines)

            # Step 4: Assemble output JSON
            output = {
                "title": outline[0]["text"] if outline else "Untitled",
                "outline": outline
            }

            # Step 5: Save to output JSON file
            save_output_json(output, output_path)


if __name__ == "__main__":
    process_all_pdfs()





    