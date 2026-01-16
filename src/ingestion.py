import os
from typing import Callable, Dict, List, Tuple
import pypdf
import docx

ExtractionResult = Tuple[str, List[str]]

def read_txt(filepath: str) -> ExtractionResult:
    """Reads a text file and returns its content as a string."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), []
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read(), ["Warning: File read with latin-1 encoding due to utf-8 error."]
        except Exception as e:
             raise ValueError(f"Error reading text file {filepath}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading text file {filepath}: {e}")

def read_pdf(filepath: str) -> ExtractionResult:
    """Extracts text from a PDF file and checks for potential ATS issues."""
    text = []
    warnings = []
    try:
        reader = pypdf.PdfReader(filepath)
        
        if reader.is_encrypted:
            warnings.append("PDF is encrypted. This might block some ATS parsers.")
            try:
                reader.decrypt("")
            except:
                pass 

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
            
            if len(page.images) > 0:
                warnings.append(f"Page {i+1}: Contains {len(page.images)} images. Text inside images is not readable by most ATS.")

        full_text = "\n".join(text)
        
        if not full_text.strip():
            warnings.append("Text extraction yielded empty result. The PDF might be an image scan.")
            
        return full_text, warnings
    except Exception as e:
        raise ValueError(f"Error reading PDF file {filepath}: {e}")

def read_docx(filepath: str) -> ExtractionResult:
    """Extracts text from a DOCX file and checks for potential ATS issues."""
    warnings = []
    try:
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        if len(doc.tables) > 0:
            warnings.append(f"Detected {len(doc.tables)} tables. Tables can cause parsing errors in older ATS systems.")
            
        if len(doc.inline_shapes) > 0:
            warnings.append(f"Detected {len(doc.inline_shapes)} inline images/shapes. Ensure these do not contain critical text.")

        return text, warnings
    except Exception as e:
        raise ValueError(f"Error reading DOCX file {filepath}: {e}")

def read_md(filepath: str) -> ExtractionResult:
    """Reads a Markdown file and returns its content as a string."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read(), []

def load_file(filepath: str) -> ExtractionResult:
    """Detects file extension and routes to the correct reader."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    readers: Dict[str, Callable[[str], ExtractionResult]] = {
        '.txt': read_txt,
        '.pdf': read_pdf,
        '.docx': read_docx,
        '.md': read_md
    }

    if os.path.isdir(filepath):
        found_file = None
        for file in os.listdir(filepath):
            ext = os.path.splitext(file)[1].lower()
            if ext in readers:
                found_file = os.path.join(filepath, file)
                break
        
        if found_file:
            print(f"Processing directory '{filepath}' - auto-selected file: {found_file}")
            filepath = found_file
        else:
            raise ValueError(f"No supported file format found in directory: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    
    if ext not in readers:
        if ext == '' and os.path.isfile(filepath):
            print(f"Warning: File '{filepath}' has no extension. Attempting to process as text file.")
            return read_txt(filepath)
            
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .txt, .pdf, .docx, .md")
        
    return readers[ext](filepath)
