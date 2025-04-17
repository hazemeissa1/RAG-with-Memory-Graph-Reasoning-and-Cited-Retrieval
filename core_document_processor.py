import os
import PyPDF2
import docx
import pandas as pd

def extract_text_from_document(file_path: str) -> str:
    """
    Extract text from supported document formats.

    Args:
        file_path: Path to document

    Returns:
        Extracted text
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == ".pdf":
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in pdf_reader.pages)
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file_extension == ".txt":
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension in [".csv", ".xlsx"]:
            df = pd.read_csv(file_path) if file_extension == ".csv" else pd.read_excel(file_path)
            return df.to_string(index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Failed to extract text: {str(e)}")
