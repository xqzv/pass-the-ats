import re
import spacy
from spacy.matcher import Matcher
from datetime import datetime
from typing import Dict, List, Optional

# Load SpaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Normalizes text using SpaCy for tokenization and filtering.
    
    Args:
        text: Input string to clean.
        remove_stopwords: Whether to remove common English stop words.
        
    Returns:
        Cleaned string with only alphanumeric characters and whitespace.
    """
    doc = nlp(text.lower())
    tokens = []
    
    for token in doc:
        # Keep tokens that are:
        # 1. Alphanumeric (is_alpha, like_num, or mix like "item1")
        # 2. Special entities looking like emails or URLs
        is_valid_token = (
            token.is_alpha or 
            token.like_num or 
            token.text.isalnum() or
            token.like_email or 
            token.like_url
        )
        
        token_text = token.text
        
        if not is_valid_token:
            # Fallback for tokens like "C++" which isn't alpha but contains alpha "C"
            # Strip non-alphanumeric chars
            cleaned = re.sub(r'[^a-zA-Z0-9]', '', token_text)
            if not cleaned:
                continue
            token_text = cleaned
            
        # Optional stopword removal
        if remove_stopwords:
            # Check original token or cleaned text
            if token.is_stop or (token_text != token.text and nlp.vocab[token_text].is_stop):
                continue
            
        tokens.append(token_text)
        
    return " ".join(tokens)

def segment_sections(text: str) -> Dict[str, str]:
    """
    Splits the text into semantic blocks using SpaCy Matcher for headers.
    """
    sections = {
        "Experience": "",
        "Education": "",
        "Skills": "",
        "Projects": "",
        "Other": ""
    }
    
    matcher = Matcher(nlp.vocab)
    
    # Define flexible patterns for headers
    # LOWER checks for case-insensitive match
    matcher.add("Experience", [[{"LOWER": {"IN": ["work", "professional", "employment"]}}, {"LOWER": "experience"}], [{"LOWER": "experience"}]])
    matcher.add("Education", [[{"LOWER": {"IN": ["education", "academic", "qualifications"]}}]])
    matcher.add("Skills", [[{"LOWER": {"IN": ["skills", "technologies", "competencies"]}}, {"OP": "*"}], [{"LOWER": "technical"}, {"LOWER": "skills"}]])
    matcher.add("Projects", [[{"LOWER": "projects"}], [{"LOWER": "personal"}, {"LOWER": "projects"}]])

    doc = nlp(text)
    
    # We identify lines that look like headers. 
    # Since Matcher works on tokens, we can scan the doc but we need to respect line breaks.
    # A simple approach is still line-by-line processing but using Matcher on lines.
    
    lines = text.split('\n')
    current_section = "Other"
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        # Process line as a doc
        line_doc = nlp(stripped_line)
        matches = matcher(line_doc)
        
        # Heuristic: If the line is short (header-like) and matches a pattern
        is_header_length = len(line_doc) <= 5
        matched_label = None
        
        if is_header_length and matches:
            # Get the match ID string
            match_id, start, end = matches[0]
            matched_label = nlp.vocab.strings[match_id]
            
        if matched_label:
            current_section = matched_label
        else:
            sections[current_section] += line + "\n"
            
    return sections

def extract_experience_years(text: str) -> float:
    """
    Estimates years of experience using SpaCy NER to find dates.
    """
    doc = nlp(text)
    
    # Extract entities labeled as DATE
    date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    
    # If generic NER misses, helpful to have a fallback or analyze the text of the entities found.
    # We still need to parse the years out of these date strings.
    
    year_pattern = r'\b(?:19|20)\d{2}\b'
    years = []
    
    # 1. Look inside identified DATE entities
    for date_str in date_entities:
        found_years = re.findall(year_pattern, date_str)
        years.extend([int(y) for y in found_years])
        
    # 2. Fallback: Scan full text if purely numeric years (like "2020") weren't caught as DATE entities
    if not years:
         years = [int(y) for y in re.findall(year_pattern, text)]
    
    if not years:
        return 0.0
        
    # Check for "Present" using token matching in the original text
    has_present = any(t.lower_ in ["present", "current"] for t in doc)
    
    min_year = min(years)
    max_year = max(years)
    
    if has_present:
        current_year = datetime.now().year
        max_year = max(max_year, current_year)
        
    return float(max_year - min_year)

def extract_key_phrases(text: str) -> List[str]:
    """
    Extracts important phrases, entities, and keywords from text using NLP.
    Prioritizes Named Entities and Noun Chunks over simple words.
    
    Args:
        text: The input text (e.g., Job Description).
        
    Returns:
        List of candidate strings (lowercase).
    """
    doc = nlp(text)
    
    allowed_pos = {'NOUN', 'PROPN', 'ADJ'}
    ignored_ent_labels = {'DATE', 'TIME', 'CARDINAL', 'QUANTITY', 'ORDINAL', 'MONEY', 'PERCENT'}
    
    # Custom blocklist for recruiting jargon and generic terms
    generic_terms = {
        "office", "downtown", "week", "month", "year", "day", "hour",
        "job", "work", "role", "candidate", "experience", "opportunity",
        "team", "company", "description", "requirement", "responsibility",
        "skill", "ability", "qualification", "preferred", "plus",
        "resume", "cv", "location", "position", "career", "employment",
        "applicant", "overview", "duty", "task", "goal", "summary",
        "benefit", "salary", "bonus", "compensation", "vision", "mission",
        "value", "culture", "environment", "growth", "start", "end",
        "good", "strong", "excellent", "great", "proficient", "knowledge",
        "looking", "seeking", "flexible", "passionate", "motivated",
        "track", "record", "proven", "successful", "detail", "oriented"
    }

    candidates = []
    
    # 1. Add Named Entities (Preserves "New York", "Microsoft Azure")
    for ent in doc.ents:
        if ent.label_ in ignored_ent_labels:
            continue
        # Clean: lowercase 
        clean_text = ent.text.strip().lower()
        if clean_text in generic_terms:
            continue
        # Don't add if it's just a generic term
        candidates.append(clean_text)

    # 2. Add Noun Chunks (Preserves "Machine Learning", "Python Developer")
    for chunk in doc.noun_chunks:
        # Simple approach: Tokenize chunk, remove stop words and generic terms
        chunk_words = [
            t.text.lower() for t in chunk 
            if not t.is_stop and t.text.lower() not in generic_terms and t.pos_ in allowed_pos
        ]
        
        if len(chunk_words) > 1: # Only add multi-word phrases here
            phrase = " ".join(chunk_words)
            candidates.append(phrase)

    # 3. Add Individual Tokens (Unigrams)
    for token in doc:
        lemma = token.lemma_.lower()
        
        if token.is_stop or lemma in generic_terms:
            continue
        if token.pos_ not in allowed_pos:
            continue
        if token.ent_type_ in ignored_ent_labels:
            continue
        if not any(c.isalnum() for c in token.text):
            continue
        if len(lemma) < 2 and lemma not in {'c', 'r', 'x', 'z'}:
            continue

        candidates.append(lemma)
        
    return candidates

