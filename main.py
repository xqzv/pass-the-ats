import argparse
import sys
from src.ingestion import load_file
from src.processing import clean_text
from src.analysis import calculate_match_score, find_missing_keywords

def main():
    parser = argparse.ArgumentParser(
        description="PassTheATS - A lightweight resume matcher."
    )
    
    parser.add_argument(
        "--resume", "-r",
        required=True,
        help="Path to the resume file (PDF, DOCX, TXT)"
    )
    
    parser.add_argument(
        "--job_desc", "-j",
        required=True,
        help="Path to the job description file (PDF, DOCX, TXT)"
    )
    
    args = parser.parse_args()
    
    # Phase 1: Ingestion
    print(f"Loading resume: {args.resume}...")
    try:
        resume_raw = load_file(args.resume)
    except FileNotFoundError:
        print(f"Error: Resume file not found at '{args.resume}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading resume: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading job description: {args.job_desc}...")
    try:
        jd_raw = load_file(args.job_desc)
    except FileNotFoundError:
        print(f"Error: Job description file not found at '{args.job_desc}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading job description: {e}", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Processing (Cleaning)
    print("Processing texts...")
    cleaned_resume = clean_text(resume_raw)
    cleaned_jd = clean_text(jd_raw)

    # Phase 3: Analysis
    print("Calculating match score...")
    score = calculate_match_score(cleaned_resume, cleaned_jd)
    missing_keywords = find_missing_keywords(cleaned_resume, cleaned_jd)
    
    # Phase 4: Output
    percentage = score * 100
    print("-" * 30)
    print(f"Match Score: {percentage:.2f}%")
    
    if missing_keywords:
        print("Missing Keywords:")
        for keyword in missing_keywords:
            print(f" - {keyword}")
    else:
        print("Great job! No key keywords are missing.")
            
    print("-" * 30)

if __name__ == "__main__":
    main()
