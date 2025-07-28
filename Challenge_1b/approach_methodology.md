# Approach Explanation

For this challenge, I built a complete offline pipeline that takes raw PDFs, figures out what parts are **most relevant based on both the persona and the job to be done**, and then **summarizes those parts accurately** — all without relying on any APIs or GPU support.

I started by **reusing my own heading detection and block extraction logic from Challenge 1a**. But I didn’t stop there. I knew this task demanded much deeper understanding of the content, so I **enhanced both the `extract_text.py` and `heading_ranker.py` scripts** to support more robust parsing. These upgrades allowed the system to better handle edge cases in heading hierarchy and paragraph structuring, especially in complex multi-language PDFs.

The input to the pipeline is a JSON file (`challenge1b_input.json`) containing a **persona** and a **job to be done**. This acts as a context filter to guide what matters most in the document.

I modularized everything into the `processing/` folder — this includes:

- `extract_text.py` – handles text extraction and heading segmentation.
- `heading_ranker.py` – assigns heading levels using context cues beyond font size.
- `section_ranker.py` – computes relevance scores between the persona+task query and each section.
- `summarizer.py` – performs extractive summarization using semantic similarity, not just keyword matching.
- `json_builder.py` – puts everything together into a neat JSON output.

**The heart of this summarization step uses the `all-MiniLM-L6-v2` model** — a small but powerful sentence transformer that turns sections and queries into embeddings. I compute cosine similarity to rank sections and choose the most relevant ones. This method is both fast and context-aware.

**To speed things up**, I implemented **multiprocessing** so that multiple documents can be processed in parallel. This is especially helpful when running everything CPU-only.

### What I’m proud of:

- **Everything is modular**, so I can plug it into future challenges or replace parts easily.
- The code is structured to handle **multi-document summarization efficiently**.
- **All logic is offline**, and works completely without cloud services.
- **Reusability and clarity** — I didn’t build a monolithic script, but a toolkit I can extend later.

If I had more time, I would have integrated a scoring system for summary quality or added multilingual support, but overall, I’m really happy with how flexible and robust the current system is.
