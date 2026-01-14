import os
from typing import Callable, Dict, List, Tuple
import pypdf
import docx

ExtractionResult = Tuple[str, List[str]]

def read_txt(filepath: str) -> ExtractionResult:
    """
    Reads a text file and returns its content as a string.
    
    Args:
        filepath: Path to the .txt file.
        
    Returns:
        A tuple containing the content of the file and a list of warnings (empty for txt).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), []
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read(), ["Warning: File read with latin-1 encoding due to utf-8 error."]
        except Exception as e:
             raise ValueError(f"Error reading text file {filepath}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading text file {filepath}: {e}")

def read_pdf(filepath: str) -> ExtractionResult:
    """
    Extracts text from a PDF file and checks for potential ATS issues.
    
    Args:
        filepath: Path to the .pdf file.
        
    Returns:
        A tuple containing extracted text and a list of warnings (e.g., images detected).
    """
    text = []
    warnings = []
    try:
        reader = pypdf.PdfReader(filepath)
        
        # Check for encryption
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
            
            # 6.2 Image/Icon Detection
            if len(page.images) > 0:
                warnings.append(f"Page {i+1}: Contains {len(page.images)} images. Text inside images is not readable by most ATS.")

        full_text = "\n".join(text)
        
        if not full_text.strip():
            warnings.append("Text extraction yielded empty result. The PDF might be an image scan.")
            
        return full_text, warnings
    except Exception as e:
        raise ValueError(f"Error reading PDF file {filepath}: {e}")

def read_docx(filepath: str) -> ExtractionResult:
    """
    Extracts text from a DOCX file and checks for potential ATS issues.
    
    Args:
        filepath: Path to the .docx file.
        
    Returns:
        A tuple containing extracted text and a list of warnings (e.g., tables detected).
    """
    warnings = []
    try:
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        # 6.1 Table Detection
        if len(doc.tables) > 0:
            warnings.append(f"Detected {len(doc.tables)} tables. Tables can cause parsing errors in older ATS systems.")
            
        # Check for images/shapes (basic check)
        if len(doc.inline_shapes) > 0:
            warnings.append(f"Detected {len(doc.inline_shapes)} inline images/shapes. Ensure these do not contain critical text.")

        return text, warnings
    except Exception as e:
        raise ValueError(f"Error reading DOCX file {filepath}: {e}")

def read_md(filepath: str) -> ExtractionResult:
    """
    Reads a Markdown file and returns its content as a string.
    
    Args:
        filepath: Path to the .md file.
        
    Returns:
         A tuple containing the content of the file and a list of warnings.
    """
    # Markdown files are text files, so we can reuse the logic or just open and read.
    # We treat it as raw text; the cleaner will handle special chars later.
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read(), []

def load_file(filepath: str) -> ExtractionResult:
    """
    Detects file extension and routes to the correct reader.
    
    Args:
        filepath: Path to the file to read.
        
    Returns:
        A tuple containing the extracted text and a list of warnings.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
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
        # Check if file has no extension
        if ext == '' and os.path.isfile(filepath):
            print(f"Warning: File '{filepath}' has no extension. Attempting to process as text file.")
            return read_txt(filepath)
            
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .txt, .pdf, .docx, .md")
        
    return readers[ext](filepath)
