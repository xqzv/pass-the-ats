import argparse
import sys
from src.ingestion import load_file
from src.processing import clean_text, segment_sections, extract_experience_years
from src.analysis import (
    calculate_match_score, 
    find_missing_keywords, 
    extract_entities,
    calculate_weighted_score,
    analyze_contextual_density,
    check_passive_voice
)

def main():
    parser = argparse.ArgumentParser(
        description="PassTheATS - A lightweight resume matcher."
    )
    
    parser.add_argument(
        "--resume", "-r",
        required=False,
        help="Path to the resume file (PDF, DOCX, TXT)"
    )
    
    parser.add_argument(
        "--job_desc", "-j",
        required=False,
        help="Path to the job description file (PDF, DOCX, TXT)"
    )
    
    args = parser.parse_args()

    if not args.resume:
        args.resume = input("Enter path to resume file: ").strip().strip('"')
    
    if not args.job_desc:
        args.job_desc = input("Enter path to job description file: ").strip().strip('"')

    
    # Phase 1: Ingestion
    print(f"Loading resume: {args.resume}...")
    try:
        resume_raw, resume_warnings = load_file(args.resume)
        if resume_warnings:
             print("\n[!] ATS Friendly Warnings for Resume:", file=sys.stderr)
             for w in resume_warnings:
                 print(f"  - {w}", file=sys.stderr)
             print("", file=sys.stderr)
             
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
        jd_raw, jd_warnings = load_file(args.job_desc)
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
    print(f"Match Score (TF-IDF): {percentage:.2f}%")
    
    # Phase 5: Structural Analysis
    print("\n--- Structural Analysis ---")
    sections = segment_sections(resume_raw)
    print(f"Sections Detected: {', '.join([k for k,v in sections.items() if v.strip()])}")
    
    weighted_score = calculate_weighted_score(sections, cleaned_jd)
    print(f"Section-Weighted Score: {weighted_score * 100:.2f}%")
    
    exp_years = extract_experience_years(sections.get("Experience", ""))
    print(f"Estimated Experience: {exp_years} years")
    
    # Phase 6: UVP Features
    print("\n--- Advanced Quality Checks ---")
    
    # Contextual Density
    density_score, broad_sentences = analyze_contextual_density(resume_raw, cleaned_jd)
    print(f"Skill Context Density: {density_score:.2f} (Skills + Action Verbs)")
    if broad_sentences:
        print("Strong Action Sentences found:")
        for sent in broad_sentences:
             print(f"  * \"{sent[:80]}...\"")
             
    # Passive Voice
    passive_pct = check_passive_voice(resume_raw)
    print(f"Passive Voice Usage: {passive_pct * 100:.1f}% (Lower is usually better)")
    
    print("-" * 30)
    if missing_keywords:
        print("Missing Keywords:")
        for keyword in missing_keywords:
            print(f" - {keyword}")
    else:
        print("Great job! No key keywords are missing.")

    # Deep NLP: Entity Extraction
    print("\n--- Extracted Entities (Deep NLP) ---")
    entities = extract_entities(resume_raw)
    important_labels = ["PERSON", "ORG", "GPE", "EDU", "DATE"]
    for label, items in entities.items():
        if items and (label in important_labels):  # Filter for relevance
             print(f"{label}: {', '.join(items[:5])}" + ("..." if len(items) > 5 else ""))
            
    print("-" * 30)

if __name__ == "__main__":
    main()
