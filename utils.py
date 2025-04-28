import fitz

def extract_pdf_text(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text") or ""
    return text