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
    """Extracts named entities from the text using spaCy."""
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
    """Calculates cosine similarity score between resume and job description using TF-IDF."""
    if not resume_text or not job_description_text:
        return 0.0

    corpus = [resume_text, job_description_text]
    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity_matrix[0][0])
    except ValueError:
        return 0.0

def find_missing_keywords(resume_text: str, jd_text: str, top_n: int | None = None) -> list[str]:
    """Identifies keywords present in JD but missing from resume."""
    if not resume_text or not jd_text:
        return []

    candidates = extract_key_phrases(jd_text)
    resume_text_lower = resume_text.lower()
    
    missing_candidates = []
    seen = set()
    
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            if cand not in resume_text_lower:
                missing_candidates.append(cand)
    
    cand_counts = Counter(candidates)
    
    final_missing = [
        (term, count) for term, count in cand_counts.most_common()
        if term in missing_candidates
    ]
    
    return [term for term, count in final_missing[:top_n]]


def calculate_weighted_score(resume_sections: Dict[str, str], jd_text: str) -> float:
    """Calculates a score based on where keywords appear in the resume."""
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

    section_weights = {
        "Experience": 2.0,
        "Projects": 1.5,
        "Skills": 1.2,
        "Education": 1.0, 
        "Other": 0.5
    }
    
    total_score = 0.0
    max_potential_score = 0.0
    
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
    Checks if technical skills (from JD) appear near Action Verbs.
    Returns normalized score and list of strong sentences.
    """
    if not resume_text:
        return 0.0, []
        
    doc = nlp(resume_text)
    jd_doc = nlp(jd_text)
    
    jd_keywords = {
        token.lemma_.lower() for token in jd_doc 
        if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN"]
    }
    
    strong_sentences = []
    
    for sent in doc.sents:
        has_verb = any(token.pos_ == "VERB" for token in sent)
        
        overlap_skills = [
            token.lemma_.lower() for token in sent 
            if token.lemma_.lower() in jd_keywords
        ]
        
        if has_verb and overlap_skills:
            strong_sentences.append(sent.text.strip())
            
    total_sentences = len(list(doc.sents))
    if total_sentences == 0:
        return 0.0, []
        
    density_score = len(strong_sentences) / total_sentences
    normalized_score = min(density_score * 5, 1.0)

    return normalized_score, strong_sentences

def check_passive_voice(text: str) -> float:
    """Calculates the percentage of sentences using passive voice."""
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

