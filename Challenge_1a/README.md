# Adobe_hackathon_2025

# Challenge 1A – Structured PDF Outline Extraction

This solution extracts **document title**, **multilingual language distribution**, and a **clean structured outline (headings with hierarchy)** from input PDFs — using a **hybrid rule-based approach** designed for robustness and speed.

## How to Run

### Prerequisites
- Python 3.9
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Docker Support 
This project also runs in a Docker container:

```bash
# Build the image
docker build -t pdf-structure .

# Run the container (input/output folders will be created automatically)
docker run -v $PWD/input:/app/input -v $PWD/output:/app/output pdf-structure
```

## Input Format

Place your PDFs in the `input/` folder.

Example:
```
Challenge_1a/
├── input/
│   └── sample.pdf
```

## Output Format

The output is saved as a JSON file in the `output/` folder for each PDF.

**Sample output:**
```json
{
  "title": "Look at the animals",
  "document_languages": ["EN", "JA"],
  "language_distribution": {
    "EN": 12,
    "JA": 10
  },
  "outline": [
    {
      "level": "H1",
      "text": "Look at the animals",
      "page": 1
    },
    {
      "level": "H3",
      "text": "Jenny Katz",
      "page": 1
    }
  ]
}
```
**Vision:**  
Many PDFs break assumptions around fonts and layouts. This solution is designed to **generalize well** without relying on predefined heading references or heavy models, enabling **real-time, multilingual outline extraction**.


## Approach

This solution uses a **hybrid, rule-based technique** with carefully engineered logic to ensure generalizability across different layouts and languages.

###  Key Features:
- **Title Detection:** Based on font size, position, patterns, and scoring logic.
- **Heading Classification:** Avoids sole reliance on font size; uses layout cues and filters to avoid body text.
- **Language Detection:** Supports multiple languages using sentence-level detection.
- **Text Preprocessing:** Merges broken lines and removes duplicates/fragments.
- **Fast & Lightweight:** Runs in under 6 seconds even on complex multilingual PDFs.


## Folder Structure

```
Challenge_1a/
├── input/               # PDF input files
├── output/              # Extracted JSON results
├── src/
│   ├── extract_text.py          # PDF parsing and text cleaning
│   ├── heading_ranker.py        # Title + heading structure logic
│   ├── language_detector.py     # Sentence-wise language tagging
│   ├── cluster_help.py          # Utilities for heading-level assignment
│   └── utils.py                 # Shared logic
├── main.py              # Entry point
├── requirements.txt
├── Dockerfile
└── README.md


## Notes
- This version focuses on accuracy and clarity using **custom rules** over ML due to real-world inconsistencies in PDF formatting.
- It gracefully handles multilingual, scanned-like, and noisy PDFs with **fallback mechanisms**.
- Easily extendable to reintegrate ML models if needed (previous versions tested SentenceTransformers).



 