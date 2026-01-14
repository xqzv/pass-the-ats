import unittest
from src.analysis import calculate_match_score, find_missing_keywords

class TestAnalysis(unittest.TestCase):

    def test_exact_match(self):
        """Test that identical texts return a score of 1.0 (or very close)."""
        text = "python developer with experience in machine learning"
        score = calculate_match_score(text, text)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_complete_mismatch(self):
        """Test that completely different texts return a score of 0.0."""
        resume = "python developer"
        jd = "nurse practitioner"
        score = calculate_match_score(resume, jd)
        # Note: Depending on TF-IDF config, there might be slight overlap if not properly cleaned,
        # but with these distinct words it should be 0.0 or very close.
        self.assertTrue(score < 0.1, f"Score {score} should be low for mismatched text")

    def test_partial_match(self):
        """Test that sharing some keywords returns a score between 0 and 1."""
        resume = "senior python developer with aws"
        jd = "junior python developer with azure"
        score = calculate_match_score(resume, jd)
        self.assertTrue(0.0 < score < 1.0)

    def test_empty_input(self):
        """Test that empty strings return 0.0."""
        self.assertEqual(calculate_match_score("", "some job"), 0.0)
        self.assertEqual(calculate_match_score("some resume", ""), 0.0)
        self.assertEqual(calculate_match_score("", ""), 0.0)

    def test_stop_words_behavior(self):
        """
        Test behavior with very common words. 
        Note: The current implementation uses default TfidfVectorizer which doesn't remove English stop words unless specified.
        So common words like 'the' might contribute to score.
        """
        resume = "the python"
        jd = "the java"
        score = calculate_match_score(resume, jd)
        self.assertTrue(score > 0.0)

    def test_find_missing_keywords(self):
        """Test that missing keywords are correctly identified."""
        resume = "experienced python developer"
        # 'java' and 'aws' are missing. 'and' is a stop word.
        jd = "experienced python developer with java and aws skills"
        
        missing = find_missing_keywords(resume, jd, top_n=5)
        
        self.assertIn("java", missing)
        self.assertIn("aws", missing)
        self.assertNotIn("python", missing)
        self.assertNotIn("developer", missing)
        self.assertNotIn("and", missing)  # Should be filtered as stop word

    def test_find_missing_keywords_ranking(self):
        """Test that missing keywords are ranked by frequency."""
        resume = "basic skills"
        # 'important' appears twice, 'rare' appears once
        jd = "important important rare skills"
        
        missing = find_missing_keywords(resume, jd, top_n=2)
        
        self.assertEqual(missing[0], "important")
        self.assertIn("rare", missing)

    def test_find_all_missing_keywords(self):
        """Test that all missing keywords are returned when top_n is None."""
        resume = "basic skills"
        jd = "item1 item2 item3 item4 item5 item6"
        
        # Should return all 6 missing items
        missing = find_missing_keywords(resume, jd)
        self.assertEqual(len(missing), 6)

if __name__ == '__main__':
    unittest.main()
