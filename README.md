# PassTheATS 

**Stop guessing why you're not getting calls. Optimize your resume offline.**

PassTheATS is a lightweight, privacy-focused CLI tool that simulates how Applicant Tracking Systems (ATS) score your resume against job descriptions.

##  Key Benefits

*   ** Privacy-First:** Your data never leaves your computer. No uploading resumes to third-party servers.
*   ** Specific & Fast:** Uses industry-standard NLP (TF-IDF & Cosine Similarity) for instant scoring.
*   ** Developer-Ready:** A clean CLI tool that fits right into your terminal workflow.
*   ** Actionable:** Get a percentage match and see exactly which keywords you are missing.

##  Upcoming Features

*   **resume.pdf** & **job.docx** parsing support.
*   Match Score % calculation.
*   Missing keyword analysis.

##  Usage

`ash
python main.py --resume "my_cv.pdf" --job "job_description.txt"
`

##  Tech Stack

*   **Python 3.10+**
*   **scikit-learn** (Vectorization)
*   **pypdf** & **python-docx** (Text Extraction)

---
*Status: In Development*
