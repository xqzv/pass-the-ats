from collections import Counter
from typing import Dict, List, Tuple
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.processing import extract_key_phrases

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli.download import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extracts named entities from the text using spaCy.
    """
    if not text:
        return {}
        
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
            
    return entities

def calculate_match_score(resume_text: str, job_description_text: str) -> float:
    """
    Calculates the cosine similarity score between a resume and a job description
    using TF-IDF vectorization.
    """
    if not resume_text or not job_description_text:
        return 0.0

    # Combine texts into a list for vectorization
    corpus = [resume_text, job_description_text]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    try:
        # Transform the texts into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        return float(similarity_matrix[0][0])
        
    except ValueError:
        return 0.0

def find_missing_keywords(resume_text: str, jd_text: str, top_n: int | None = None) -> list[str]:
    """
    Identifies keywords present in the job description but missing from the resume
    using extract_key_phrases for extraction and plain text search for matching.
    """
    if not resume_text or not jd_text:
        return []

    # Extract key phrases from JD using the processing module
    candidates = extract_key_phrases(jd_text)
    
    # Find missing
    # We check if candidate string is present in resume text (lowercase)
    resume_text_lower = resume_text.lower()
    
    missing_candidates = []
    seen = set()
    
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            # Check presence
            if cand not in resume_text_lower:
                missing_candidates.append(cand)
    
    cand_counts = Counter(candidates)
    
    # Filter missing from counts
    final_missing = [
        (term, count) for term, count in cand_counts.most_common()
        if term in missing_candidates
    ]
    
    # Return top N terms
    return [term for term, count in final_missing[:top_n]]


# ---------------------------------------------------------
# Phase 5 & 6: Advanced Analysis
# ---------------------------------------------------------

def calculate_weighted_score(resume_sections: Dict[str, str], jd_text: str) -> float:
    """
    Calculates a score based on where keywords appear in the resume.
    """
    # 1. Identify important keywords from JD
    if not jd_text:
        return 0.0
        
    jd_doc = nlp(jd_text)
    jd_words = [
        token.lemma_.lower() for token in jd_doc 
        if (token.is_alpha or token.like_num or token.text.isalnum()) and not token.is_stop
    ]
    
    if not jd_words:
        return 0.0
        
    jd_counter = Counter(jd_words)
    top_keywords = [w for w, _ in jd_counter.most_common(20)]
    
    if not top_keywords:
        return 0.0

    # 2. Define weights
    section_weights = {
        "Experience": 2.0,
        "Projects": 1.5,
        "Skills": 1.2,
        "Education": 1.0, 
        "Other": 0.5
    }
    
    # 3. Calculate score
    total_score = 0.0
    max_potential_score = 0.0
    
    # Pre-process sections
    processed_sections = {k: v.lower() for k, v in resume_sections.items()}
    
    for kw in top_keywords:
        max_potential_score += max(section_weights.values())
        
        found_weight = 0.0
        for section, content in processed_sections.items():
            if kw in content:
                weight = section_weights.get(section, 0.5)
                if weight > found_weight:
                    found_weight = weight
        
        total_score += found_weight
        
    if max_potential_score == 0:
        return 0.0
        
    return total_score / max_potential_score

def analyze_contextual_density(resume_text: str, jd_text: str) -> Tuple[float, List[str]]:
    """
    UVP 2: Checks if technical skills (from JD) appear near Action Verbs.
    Uses SpaCy POS tagging to identify verbs dynamically.
    """
    if not resume_text:
        return 0.0, []
        
    doc = nlp(resume_text)
    jd_doc = nlp(jd_text)
    
    # Extract potential skills from JD (nouns/proper nouns that are not stop words)
    jd_keywords = {
        token.lemma_.lower() for token in jd_doc 
        if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN"]
    }
    
    strong_sentences = []
    
    for sent in doc.sents:
        # Check for action verb in the sentence
        has_verb = any(token.pos_ == "VERB" for token in sent)
        
        # Check for JD keywords in the same sentence
        overlap_skills = [
            token.lemma_.lower() for token in sent 
            if token.lemma_.lower() in jd_keywords
        ]
        
        if has_verb and overlap_skills:
            strong_sentences.append(sent.text.strip())
            
    # Score based on ratio of strong sentences to total sentences
    # capped at some reasonable limit
    total_sentences = len(list(doc.sents))
    if total_sentences == 0:
        return 0.0, []
        
    density_score = len(strong_sentences) / total_sentences
    # Normalized: if 20% of sentences are strong, that's great (1.0).
    normalized_score = min(density_score * 5, 1.0)

    
    return normalized_score, strong_sentences

def check_passive_voice(text: str) -> float:
    """
    Calculates the percentage of sentences using passive voice.
    Lower is better for resumes.
    """
    if not text:
        return 0.0
        
    doc = nlp(text)
    passive_sentences = 0
    total_sentences = 0
    
    for sent in doc.sents:
        total_sentences += 1
        for token in sent:
            if token.dep_ == "nsubjpass" or token.dep_ == "auxpass":
                passive_sentences += 1
                break
                
    if total_sentences == 0:
        return 0.0
        
    return passive_sentences / total_sentences

