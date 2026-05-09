"""Resume and job description parsing utilities."""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document
except ImportError:
    Document = None

from preprocessing import NLP, normalize_text, tokenize


SKILL_LIBRARY = {
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "html",
    "css", "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring", "streamlit", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "keras", "machine learning", "deep learning", "nlp", "computer vision",
    "data analysis", "data visualization", "power bi", "tableau", "excel",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "rest api",
    "mongodb", "postgresql", "mysql", "spark", "hadoop", "airflow", "etl",
    "statistics", "a/b testing", "figma", "ui/ux", "communication", "leadership",
    "problem solving", "agile", "scrum", "ci/cd",
}

SECTION_HEADERS = {
    "education": ["education", "academic background", "qualification"],
    "experience": ["experience", "work experience", "employment", "professional experience"],
    "projects": ["projects", "academic projects", "personal projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "skills": ["skills", "technical skills", "core skills"],
}


@dataclass
class ParsedProfile:
    """Structured representation of one candidate resume."""
    file_name: str
    raw_text: str
    name: str = "Unknown Candidate"
    email: str = ""
    phone: str = ""
    skills: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    experience: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "file_name": self.file_name,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "skills": self.skills,
            "education": self.education,
            "experience": self.experience,
            "projects": self.projects,
            "certifications": self.certifications,
            "raw_text": self.raw_text,
        }


def extract_text_from_file(file) -> str:
    """Extract text from an uploaded PDF, DOCX, TXT, or Streamlit file object."""
    suffix = Path(file.name).suffix.lower()

    if suffix == ".pdf":
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF parsing.")
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return normalize_document_text("\n".join(text_parts))

    if suffix == ".docx":
        if Document is None:
            raise ImportError("python-docx is required for DOCX parsing.")
        doc = Document(file)
        return normalize_document_text("\n".join(paragraph.text for paragraph in doc.paragraphs))

    raw = file.read()
    if isinstance(raw, bytes):
        return normalize_document_text(raw.decode("utf-8", errors="ignore"))
    return normalize_document_text(raw)


def parse_resume_text(text: str, file_name: str = "resume.txt") -> ParsedProfile:
    """Parse resume text into structured candidate data."""
    text = normalize_document_text(text)
    profile = ParsedProfile(file_name=file_name, raw_text=text)
    profile.email = extract_email(text)
    profile.phone = extract_phone(text)
    profile.name = extract_name(text, profile.email)
    sections = split_sections(text)

    profile.skills = extract_skills(text)
    profile.education = extract_section_lines(sections, "education")
    profile.experience = extract_section_lines(sections, "experience")
    profile.projects = extract_section_lines(sections, "projects")
    profile.certifications = extract_section_lines(sections, "certifications")
    return profile


def normalize_document_text(text: str) -> str:
    """Clean document text while preserving useful line breaks."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_uploaded_resume(file) -> ParsedProfile:
    """Parse one uploaded resume file."""
    text = extract_text_from_file(file)
    return parse_resume_text(text, file.name)


def extract_email(text: str) -> str:
    match = re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)
    return match.group(0).strip() if match else ""


def extract_name(text: str, email: str = "") -> str:
    """Use spaCy PERSON entities when possible, otherwise use first resume line."""
    if NLP is not None and "ner" in NLP.pipe_names:
        doc = NLP(text[:1000])
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
                return ent.text.strip()

    lines = [line.strip() for line in re.split(r"[\n\r]+", text) if line.strip()]
    for line in lines[:6]:
        if email and email in line:
            continue
        if not re.search(r"@|http|www|resume|curriculum", line, flags=re.I):
            words = line.split()
            if 1 < len(words) <= 5:
                return line
    return "Unknown Candidate"


def split_sections(text: str) -> Dict[str, List[str]]:
    """Group resume lines under common resume section headings."""
    sections = {section: [] for section in SECTION_HEADERS}
    current_section: Optional[str] = None

    for raw_line in re.split(r"[\n\r]+|(?<=\.)\s{2,}", text):
        line = raw_line.strip(" -:\t")
        if not line:
            continue
        lower_line = line.lower()
        matched_section = None
        for section, aliases in SECTION_HEADERS.items():
            if lower_line in aliases or any(lower_line.startswith(alias + ":") for alias in aliases):
                matched_section = section
                break
        if matched_section:
            current_section = matched_section
            continue
        if current_section:
            sections[current_section].append(line)
    return sections


def extract_section_lines(sections: Dict[str, List[str]], section: str, limit: int = 8) -> List[str]:
    """Return compact section lines for display and generation."""
    return [line for line in sections.get(section, [])[:limit] if len(line) > 2]


def extract_skills(text: str) -> List[str]:
    """Extract skills using a curated skill library and token matching."""
    normalized = f" {text.lower()} "
    found = set()
    for skill in SKILL_LIBRARY:
        pattern = r"(?<![a-zA-Z0-9+#])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9+#])"
        if re.search(pattern, normalized):
            found.add(skill)
    return sorted(found)


def analyze_job_description(text: str) -> Dict:
    """Extract skills, keywords, experience level, and role category from a JD."""
    text = normalize_text(text)
    skills = extract_skills(text)
    keywords = extract_keywords(text)
    experience_level = extract_experience_level(text)
    role_category = infer_role_category(text, skills)
    return {
        "raw_text": text,
        "required_skills": skills,
        "keywords": keywords,
        "experience_level": experience_level,
        "role_category": role_category,
    }


def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    words = [word.lower() for word in tokenize(text) if len(word) > 2]
    stop = {"and", "the", "with", "for", "you", "our", "are", "will", "this", "that"}
    counts = Counter(word for word in words if word not in stop)
    return [word for word, _ in counts.most_common(top_n)]


def extract_experience_level(text: str) -> str:
    lower = text.lower()
    years = [int(value) for value in re.findall(r"(\d+)\+?\s*(?:years|yrs)", lower)]
    if years:
        max_years = max(years)
        if max_years >= 7:
            return "Senior"
        if max_years >= 3:
            return "Mid-level"
        return "Entry-level"
    if any(term in lower for term in ["senior", "lead", "principal", "manager"]):
        return "Senior"
    if any(term in lower for term in ["intern", "junior", "entry"]):
        return "Entry-level"
    return "Mid-level"


def infer_role_category(text: str, skills: List[str]) -> str:
    lower = text.lower()
    skill_set = set(skills)
    if any(term in lower for term in ["data scientist", "machine learning", "ml engineer"]) or {"python", "machine learning", "statistics"} & skill_set:
        return "Data Science"
    if any(term in lower for term in ["frontend", "front-end", "react", "ui developer"]) or {"react", "javascript", "html", "css"} & skill_set:
        return "Frontend Development"
    if any(term in lower for term in ["backend", "api", "django", "fastapi", "spring"]) or {"django", "fastapi", "node.js", "sql"} & skill_set:
        return "Backend Development"
    if any(term in lower for term in ["devops", "cloud", "kubernetes", "docker"]) or {"aws", "docker", "kubernetes"} & skill_set:
        return "Cloud/DevOps"
    return "General Technology"
