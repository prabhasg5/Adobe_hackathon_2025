# 🔍 Challenge 1B: Persona-Based PDF Summarizer (Offline)

This project solves the Adobe Hackathon Challenge 1B — **summarizing sections of a PDF based on a given persona and job-to-be-done**, completely **offline** and **modular**. The system processes real PDFs, ranks sections by relevance, and summarizes only what matters.

---

## 🧠 What it Does

- Takes in **multiple PDFs** and a JSON query with:
  - A **persona**
  - A **job to be done**

- Outputs a structured JSON that includes:
  - PDF metadata
  - Top-ranked relevant sections
  - Extractive summaries for each section
  - Subsection-wise analysis

- Works offline using **sentence-transformers (`all-MiniLM-L6-v2`)**
- Uses **parallel processing** to speed up multi-document handling
- Fully modular and reusable components

---

## 🗂️ Repo Structure

```
Challenge_1b/
├── main.py                     # Orchestrates the entire flow
├── challenge1b_input.json      # Sample input file
├── challenge1b_output.json     # Sample output
├── processing/
│   ├── extract_text.py         # PDF text + heading extractor
│   ├── section_ranker.py       # Matches persona+job to sections
│   ├── summarizer.py           # Extractive summarization logic
│   ├── json_builder.py         # Final JSON output builder
│   └── ...
├── approach_explanation.md     # 300-500 word approach write-up
├── README.md                   # You're here!
└── Dockerfile                  # (Optional) Docker environment setup
```

---

## 🚀 How to Run

> 🔧 Requirements: Python 3.9+, `pip`, and optionally `Docker`.

### 1. Install requirements:

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline:

```bash
python main.py --input challenge1b_input.json
```

This will generate `challenge1b_output.json` with all ranked sections and summaries.

---

## 🐳 Docker Setup (Optional)

Build and run the container:

```bash
docker build -t pdf-summarizer .
docker run -v $(pwd):/app pdf-summarizer
```

---

## ✅ Features Recap

- ✅ Offline, fast, and accurate
- ✅ Designed for real-world PDFs
- ✅ Persona-aware section filtering
- ✅ Built-in multiprocessing
- ✅ Human-like approach explanation included!

---

## 📄 Sample Input Format

```json
{
  "persona": "A product manager new to generative AI",
  "job_to_be_done": "Understand ethical risks of using AI tools",
  "documents": [
    "data/sample_doc1.pdf",
    "data/sample_doc2.pdf"
  ]
}
```

---

## 👨‍💻 Author

Built with love and lots of thought by **M. Jayananda Prabhas**