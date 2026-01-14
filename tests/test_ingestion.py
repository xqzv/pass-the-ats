import unittest
import os
import sys

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_file, read_txt

class TestIngestion(unittest.TestCase):
    def setUp(self):
        self.test_txt = "test_resume.txt"
        with open(self.test_txt, "w") as f:
            f.write("Hello World")

    def tearDown(self):
        if os.path.exists(self.test_txt):
            os.remove(self.test_txt)

    def test_read_txt(self):
        content, warnings = read_txt(self.test_txt)
        self.assertEqual(content, "Hello World")
        self.assertEqual(warnings, [])

    def test_load_file_txt(self):
        content, warnings = load_file(self.test_txt)
        self.assertEqual(content, "Hello World")
        self.assertEqual(warnings, [])

    def test_load_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_file("nonexistent.txt")

    def test_load_file_unsupported(self):
        dummy_file = "test.xyz"
        with open(dummy_file, "w") as f:
            f.write("data")
        try:
            with self.assertRaises(ValueError):
                load_file(dummy_file)
        finally:
            if os.path.exists(dummy_file):
                os.remove(dummy_file)

if __name__ == '__main__':
    unittest.main()
