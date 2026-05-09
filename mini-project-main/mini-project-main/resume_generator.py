"""Personalized resume generation and export utilities."""

from io import BytesIO
from typing import Dict, List

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
except ImportError:
    LETTER = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None

from resume_parser import ParsedProfile, analyze_job_description


def generate_summary(profile: ParsedProfile, job_analysis: Dict) -> str:
    """Create a truthful, JD-targeted resume summary."""
    relevant_skills = [skill for skill in profile.skills if skill in job_analysis["required_skills"]]
    top_skills = ", ".join(relevant_skills[:6] or profile.skills[:6])
    role = job_analysis["role_category"]
    if top_skills:
        return (
            f"{profile.name} is a {role.lower()} candidate with hands-on exposure to "
            f"{top_skills}. The profile is tailored for this role by emphasizing relevant "
            "technical skills, project work, and measurable contributions already present "
            "in the candidate resume."
        )
    return (
        f"{profile.name} is a motivated candidate for a {role.lower()} role. This resume "
        "version highlights relevant education, projects, and experience found in the "
        "uploaded resume."
    )


def reorder_by_relevance(items: List[str], keywords: List[str]) -> List[str]:
    """Sort resume bullets so JD-relevant content appears first."""
    keyword_set = {keyword.lower() for keyword in keywords}

    def score(item: str) -> int:
        lower = item.lower()
        return sum(1 for keyword in keyword_set if keyword in lower)

    return sorted(items, key=score, reverse=True)


def generate_improvement_suggestions(profile: ParsedProfile, match_result: Dict) -> List[str]:
    """Generate honest resume improvement suggestions without inventing facts."""
    suggestions = []
    if match_result["missing_skills"]:
        suggestions.append(
            "Add truthful examples or coursework for these missing skills if you have them: "
            + ", ".join(match_result["missing_skills"][:8])
        )
    if not profile.projects:
        suggestions.append("Add 2-3 relevant projects with tools used, problem solved, and measurable results.")
    if not profile.certifications:
        suggestions.append("Add relevant certifications or training if available.")
    suggestions.append("Use job-description keywords naturally in project and experience bullets.")
    suggestions.append("Quantify impact with metrics such as accuracy, time saved, revenue, users, or scale.")
    return suggestions


def generate_resume(profile: ParsedProfile, job_description: str, match_result: Dict, template: str = "Modern") -> str:
    """Generate an editable, optimized resume in Markdown format."""
    job_analysis = analyze_job_description(job_description)
    relevant_skills = reorder_by_relevance(profile.skills, job_analysis["required_skills"] + job_analysis["keywords"])
    projects = reorder_by_relevance(profile.projects, job_analysis["keywords"])
    experience = reorder_by_relevance(profile.experience, job_analysis["keywords"])
    suggestions = generate_improvement_suggestions(profile, match_result)

    divider = "---" if template == "Modern" else ""
    title = f"# {profile.name}" if template != "Compact" else f"## {profile.name}"

    lines = [
        title,
        f"{profile.email} | {profile.phone}".strip(" |"),
        divider,
        "## Targeted Summary",
        generate_summary(profile, job_analysis),
        "## Relevant Skills",
        ", ".join(relevant_skills) if relevant_skills else "Add relevant verified skills here.",
        "## Experience",
    ]

    lines.extend(format_bullets(experience, "Add relevant professional experience here."))
    lines.append("## Projects")
    lines.extend(format_bullets(projects, "Add relevant projects here."))
    lines.append("## Education")
    lines.extend(format_bullets(profile.education, "Add education details here."))
    lines.append("## Certifications")
    lines.extend(format_bullets(profile.certifications, "Add certifications here."))
    lines.append("## Optimization Notes")
    lines.extend(f"- {suggestion}" for suggestion in suggestions)

    return "\n\n".join(line for line in lines if line is not None)


def format_bullets(items: List[str], fallback: str) -> List[str]:
    """Format resume section content as Markdown bullets."""
    if not items:
        return [f"- {fallback}"]
    return [f"- {item}" for item in items[:8]]


def markdown_to_pdf(markdown_text: str) -> bytes:
    """Export generated resume text to a simple PDF."""
    if SimpleDocTemplate is None:
        raise ImportError("reportlab is required for PDF export.")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = []

    for line in markdown_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line.replace("# ", ""), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line.replace("## ", ""), styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph(line, styles["BodyText"]))
        elif line == "---":
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(line, styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()


def generate_interview_questions(match_result: Dict) -> List[str]:
    """Generate interview questions from matched and missing skills."""
    skills = match_result.get("matched_skills", [])[:5] + match_result.get("missing_skills", [])[:3]
    if not skills:
        skills = match_result.get("job_keywords", [])[:5]
    return [f"Can you explain a project or experience where you used {skill}?" for skill in skills]

