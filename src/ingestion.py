import os
from typing import Callable, Dict
import pypdf
import docx

def read_txt(filepath: str) -> str:
    """
    Reads a text file and returns its content as a string.
    
    Args:
        filepath: Path to the .txt file.
        
    Returns:
        The content of the file as a string.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(filepath: str) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        filepath: Path to the .pdf file.
        
    Returns:
        The extracted text as a string.
    """
    text = []
    try:
        reader = pypdf.PdfReader(filepath)
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        raise ValueError(f"Error reading PDF file {filepath}: {e}")

def read_docx(filepath: str) -> str:
    """
    Extracts text from a DOCX file.
    
    Args:
        filepath: Path to the .docx file.
        
    Returns:
        The extracted text as a string.
    """
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"Error reading DOCX file {filepath}: {e}")

def load_file(filepath: str) -> str:
    """
    Detects file extension and routes to the correct reader.
    
    Args:
        filepath: Path to the file to read.
        
    Returns:
        The content of the file as a string.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    
    readers: Dict[str, Callable[[str], str]] = {
        '.txt': read_txt,
        '.pdf': read_pdf,
        '.docx': read_docx
    }
    
    if ext not in readers:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .txt, .pdf, .docx")
        
    return readers[ext](filepath)
