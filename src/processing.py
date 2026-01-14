import re

# Basic list of English stop words
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", 
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", 
    "to", "was", "were", "will", "with", "this", "but", "they", "have",
    "had", "what", "when", "where", "who", "which", "why", "not", "no", 
    "yes", "can", "you", "your", "my", "mine", "ours", "our", "him", 
    "her", "his", "she", "he", "it", "we", "they"
}

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Normalizes text by converting to lowercase and removing special characters.
    
    Args:
        text: Input string to clean.
        remove_stopwords: Whether to remove common English stop words.
        
    Returns:
        Cleaned string with only alphanumeric characters and whitespace.
    """
    # 2.1: Convert all text to lowercase.
    text = text.lower()
    
    # 2.2: Remove special characters/punctuation
    # Pattern matches any character that is NOT alphanumeric or whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 2.3: Remove stop words
    if remove_stopwords:
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
        text = " ".join(filtered_tokens)
    
    return text
