import unittest
from src.processing import clean_text

class TestProcessing(unittest.TestCase):
    def test_lowercase(self):
        """Test that text is converted to lowercase."""
        self.assertEqual(clean_text("Hello World"), "hello world")
        self.assertEqual(clean_text("PYTHON"), "python")

    def test_remove_punctuation(self):
        """Test that non-alphanumeric characters are removed."""
        self.assertEqual(clean_text("Hello, World!"), "hello world")
        self.assertEqual(clean_text("Email: test@example.com"), "email testexamplecom")
        self.assertEqual(clean_text("C++"), "c")

    def test_numbers_preserved(self):
        """Test that numbers are preserved."""
        self.assertEqual(clean_text("Python 3.10"), "python 310")
        self.assertEqual(clean_text("Year 2023"), "year 2023")

    def test_whitespace_preserved(self):
        """Test that whitespace (spaces, newlines, tabs) is preserved."""
        text = "Line 1\nLine 2\tTabbed"
        expected = "line 1\nline 2\ttabbed"
        self.assertEqual(clean_text(text), expected)

    def test_empty_string(self):
        """Test handling of empty strings."""
        self.assertEqual(clean_text(""), "")

    def test_remove_stopwords(self):
        """Test that stop words are removed when requested."""
        text = "This is a test of the system."
        # Without flag (default) - stopwords remain
        # Note: clean_text lowercases everything first
        self.assertEqual(clean_text(text), "this is a test of the system")
        
        # With flag
        # Expected: "test system" (assuming "this", "is", "a", "of", "the" are in STOP_WORDS)
        expected = "test system" 
        self.assertEqual(clean_text(text, remove_stopwords=True), expected)

if __name__ == '__main__':
    unittest.main()
