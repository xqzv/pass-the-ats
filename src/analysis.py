from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.processing import STOP_WORDS

def calculate_match_score(resume_text: str, job_description_text: str) -> float:
    """
    Calculates the cosine similarity score between a resume and a job description
    using TF-IDF vectorization.

    Args:
        resume_text (str): The preprocessed text content of the resume.
        job_description_text (str): The preprocessed text content of the job description.

    Returns:
        float: A similarity score between 0.0 and 1.0 (1.0 being an exact match).
               Returns 0.0 if either input is empty.
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
        # tfidf_matrix is a 2xN matrix. We want similarity between row 0 and row 1.
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        # The result is a 1x1 matrix containing the score
        score = similarity_matrix[0][0]
        
        return float(score)
        
    except ValueError:
        # Handle cases where vectorization might fail (e.g. stop words removal resulted in empty strings)
        return 0.0

def find_missing_keywords(resume_text: str, jd_text: str, top_n: int | None = None) -> list[str]:
    """
    Identifies keywords present in the job description but missing from the resume.
    
    Args:
        resume_text (str): Preprocessed resume text.
        jd_text (str): Preprocessed job description text.
        top_n (int | None): Number of missing keywords to return (ranked by frequency in JD). 
                            If None, returns all missing keywords.
        
    Returns:
        list[str]: List of missing keywords.
    """
    if not resume_text or not jd_text:
        return []

    resume_words = set(resume_text.split())
    jd_words = jd_text.split()
    
    # Filter stopwords and words already in resume
    # We check if word is in STOP_WORDS. valid words are those NOT in STOP_WORDS
    missing_candidates = [
        word for word in jd_words 
        if word not in STOP_WORDS and word not in resume_words
    ]
    
    # Count frequency of missing words in JD to find the most "important" ones
    # (Assumption: more frequent mention = more important)
    counter = Counter(missing_candidates)
    
    # Return top N most frequent missing words
    return [word for word, count in counter.most_common(top_n)]

