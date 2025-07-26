import fitz  

def extract_text_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    font_sizes = [span["size"] for span in line["spans"]]
                    data.append({
                        "text": line_text.strip(),
                        "page": page_num + 1,
                        "font_size": max(font_sizes) if font_sizes else 0
                    })
    return data




