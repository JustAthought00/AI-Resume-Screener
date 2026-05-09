"""Resume to job matching engine using TF-IDF and optional embeddings."""

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text, keyword_overlap
from resume_parser import ParsedProfile, analyze_job_description

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def tfidf_similarity(resume_text: str, job_text: str) -> float:
    """Calculate cosine similarity between resume and JD using TF-IDF."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
    matrix = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_text)])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def embedding_similarity(resume_text: str, job_text: str) -> float:
    """Calculate semantic similarity using sentence-transformers when installed."""
    if SentenceTransformer is None:
        return 0.0
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([resume_text, job_text])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    except Exception:
        return 0.0


def skill_match(candidate_skills: List[str], required_skills: List[str]) -> Dict:
    """Compare candidate skills with required job skills."""
    candidate = {skill.lower() for skill in candidate_skills}
    required = {skill.lower() for skill in required_skills}
    matched = sorted(candidate.intersection(required))
    missing = sorted(required.difference(candidate))
    percentage = len(matched) / len(required) if required else 0.0
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_match_percentage": percentage,
    }


def calculate_match(profile: ParsedProfile, job_description: str, use_embeddings: bool = False) -> Dict:
    """Generate all matching scores for one candidate."""
    jd = analyze_job_description(job_description)
    tfidf_score = tfidf_similarity(profile.raw_text, job_description)
    embed_score = embedding_similarity(profile.raw_text, job_description) if use_embeddings else 0.0
    skills = skill_match(profile.skills, jd["required_skills"])
    missing_keywords = keyword_overlap(profile.raw_text.lower().split(), jd["keywords"])

    semantic_score = embed_score if embed_score > 0 else tfidf_score
    ats_score = (
        0.45 * tfidf_score
        + 0.35 * skills["skill_match_percentage"]
        + 0.20 * semantic_score
    )
    ats_score = float(np.clip(ats_score, 0, 1))

    return {
        "candidate": profile.name,
        "file_name": profile.file_name,
        "email": profile.email,
        "phone": profile.phone,
        "ats_score": round(ats_score * 100, 2),
        "tfidf_similarity": round(tfidf_score * 100, 2),
        "embedding_similarity": round(embed_score * 100, 2) if embed_score else 0.0,
        "skill_match_percentage": round(skills["skill_match_percentage"] * 100, 2),
        "matched_skills": skills["matched_skills"],
        "missing_skills": skills["missing_skills"],
        "missing_keywords": missing_keywords[:15],
        "experience_level": jd["experience_level"],
        "role_category": jd["role_category"],
        "required_skills": jd["required_skills"],
        "job_keywords": jd["keywords"],
    }


def rank_candidates(profiles: List[ParsedProfile], job_description: str, use_embeddings: bool = False) -> pd.DataFrame:
    """Rank multiple candidates against one job description."""
    rows = [calculate_match(profile, job_description, use_embeddings) for profile in profiles]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ats_score", ascending=False).reset_index(drop=True)


def save_rankings(ranked_df: pd.DataFrame, path: str = "models/latest_rankings.joblib") -> None:
    """Persist the latest ranking table for audit or later review."""
    output_path = Path(path)
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(ranked_df, output_path)


def load_rankings(path: str = "models/latest_rankings.joblib") -> pd.DataFrame:
    """Load saved candidate rankings if available."""
    ranking_path = Path(path)
    if not ranking_path.exists():
        return pd.DataFrame()
    return joblib.load(ranking_path)


def recommend_skills(missing_skills: List[str], role_category: str) -> List[str]:
    """Recommend practical skills to learn based on missing skills and role."""
    role_defaults = {
        "Data Science": ["python", "sql", "machine learning", "statistics", "data visualization"],
        "Frontend Development": ["javascript", "react", "html", "css", "ui/ux"],
        "Backend Development": ["python", "sql", "rest api", "docker", "cloud"],
        "Cloud/DevOps": ["aws", "docker", "kubernetes", "ci/cd", "linux"],
    }
    suggestions = list(dict.fromkeys(missing_skills + role_defaults.get(role_category, [])))
    return suggestions[:8]
