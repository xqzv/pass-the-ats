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
    """Normalizes text using SpaCy for tokenization and filtering."""
    doc = nlp(text.lower())
    tokens = []
    
    for token in doc:
        is_valid_token = (
            token.is_alpha or 
            token.like_num or 
            token.text.isalnum() or
            token.like_email or 
            token.like_url
        )
        
        token_text = token.text
        
        if not is_valid_token:
            cleaned = re.sub(r'[^a-zA-Z0-9]', '', token_text)
            if not cleaned:
                continue
            token_text = cleaned
            
        if remove_stopwords:
            if token.is_stop or (token_text != token.text and nlp.vocab[token_text].is_stop):
                continue
            
        tokens.append(token_text)
        
    return " ".join(tokens)

def segment_sections(text: str) -> Dict[str, str]:
    """Splits the text into semantic blocks using SpaCy Matcher for headers."""
    sections = {
        "Experience": "",
        "Education": "",
        "Skills": "",
        "Projects": "",
        "Other": ""
    }
    
    matcher = Matcher(nlp.vocab)
    
    matcher.add("Experience", [[{"LOWER": {"IN": ["work", "professional", "employment"]}}, {"LOWER": "experience"}], [{"LOWER": "experience"}]])
    matcher.add("Education", [[{"LOWER": {"IN": ["education", "academic", "qualifications"]}}]])
    matcher.add("Skills", [[{"LOWER": {"IN": ["skills", "technologies", "competencies"]}}, {"OP": "*"}], [{"LOWER": "technical"}, {"LOWER": "skills"}]])
    matcher.add("Projects", [[{"LOWER": "projects"}], [{"LOWER": "personal"}, {"LOWER": "projects"}]])

    doc = nlp(text)
    
    lines = text.split('\n')
    current_section = "Other"
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        line_doc = nlp(stripped_line)
        matches = matcher(line_doc)
        
        is_header_length = len(line_doc) <= 5
        matched_label = None
        
        if is_header_length and matches:
            match_id, start, end = matches[0]
            matched_label = nlp.vocab.strings[match_id]
            
        if matched_label:
            current_section = matched_label
        else:
            sections[current_section] += line + "\n"
            
    return sections

def extract_experience_years(text: str) -> float:
    """Estimates years of experience using SpaCy NER to find dates."""
    doc = nlp(text)
    
    date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    
    year_pattern = r'\b(?:19|20)\d{2}\b'
    years = []
    
    for date_str in date_entities:
        found_years = re.findall(year_pattern, date_str)
        years.extend([int(y) for y in found_years])
        
    if not years:
         years = [int(y) for y in re.findall(year_pattern, text)]
    
    if not years:
        return 0.0
        
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
    """
    doc = nlp(text)
    
    allowed_pos = {'NOUN', 'PROPN'} # Removed ADJ for single tokens to reduce noise
    ignored_ent_labels = {'DATE', 'TIME', 'CARDINAL', 'QUANTITY', 'ORDINAL', 'MONEY', 'PERCENT', 'LAW'}
    
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
        "track", "record", "proven", "successful", "detail", "oriented",
        "clean", "maintainable", "efficient", "collaborate", "cross",
        "functional", "design", "discussion", "development", "proficiency",
        "familiarity", "nice", "understanding", "concept", "system",
        "technology", "platform", "communication"
    }

    candidates = []
    
    for ent in doc.ents:
        if ent.label_ in ignored_ent_labels:
            continue
        clean_text = ent.text.strip().lower()
        if clean_text in generic_terms:
            continue
        candidates.append(clean_text)

    chunk_allowed_pos = {'NOUN', 'PROPN', 'ADJ'}

    for chunk in doc.noun_chunks:
        chunk_words = [
            t.text.lower() for t in chunk 
            if not t.is_stop and t.text.lower() not in generic_terms and t.pos_ in chunk_allowed_pos
        ]
        
        if len(chunk_words) > 1:
            phrase = " ".join(chunk_words)
            # Filter out phrases that are just a string of adjectives
            if any(t.pos_ in {'NOUN', 'PROPN'} for t in chunk if t.text.lower() in phrase):
                 candidates.append(phrase)

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

