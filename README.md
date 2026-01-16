# NLP Skill Matcher

**Quantify the relevance of your resume to job descriptions using local NLP analysis.**

This project is a lightweight, privacy-focused CLI tool designed to help candidates tailor their applications. It uses Natural Language Processing (TF-IDF & Cosine Similarity) to analyze how well a resume aligns with specific job requirements.

## Key Benefits

*   **Privacy-First:** Data processing happens locally. No personal documents are uploaded to external servers.
*   **Transparent Metrics:** Uses standard NLP techniques to provide an objective relevance score.
*   **Developer-Ready:** A clean CLI tool that integrates into a terminal workflow.
*   **Insightful:** Identifies potential skill gaps by comparing vocabulary between documents.

## Upcoming Features

*   **resume.pdf** & **job.docx** parsing support.
*   Relevance Score % calculation.
*   Skill gap analysis.

## Usage

```bash
python main.py --resume "my_cv.pdf" --job "job_description.txt"
```

## Tech Stack

*   **Python 3.10+**
*   **scikit-learn** (Vectorization)
*   **pypdf** & **python-docx** (Text Extraction)

---
*Status: In Development*
