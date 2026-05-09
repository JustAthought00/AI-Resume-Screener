"""Streamlit app for AI-powered resume screening and job matching."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from job_matcher import calculate_match, rank_candidates, recommend_skills, save_rankings
from resume_generator import generate_interview_questions, generate_resume, markdown_to_pdf
from resume_parser import analyze_job_description, extract_text_from_file, parse_resume_text, parse_uploaded_resume


ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_JD_PATH = ROOT_DIR / "datasets" / "sample_job_description.txt"
SAMPLE_RESUME_PATH = ROOT_DIR / "datasets" / "sample_resume.txt"


st.set_page_config(page_title="Resume Screening and Job Matching", page_icon="RS", layout="wide")


def load_sample_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def render_score(label: str, value: float):
    st.metric(label, f"{value:.2f}%")


def plot_skill_frequency(ranked_df: pd.DataFrame):
    all_skills = []
    for skills in ranked_df.get("matched_skills", []):
        all_skills.extend(skills)
    if not all_skills:
        st.info("No matched skills available for the chart yet.")
        return
    counts = pd.Series(all_skills).value_counts().head(12)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#2f80ed")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Skill")
    ax.set_title("Top Matched Skills")
    st.pyplot(fig)


def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.8rem;}
        .hero-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem;}
        .muted {color: #5c6670;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='hero-title'>AI Resume Screening and Job Matching</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='muted'>Parse resumes, compare them with a job description, rank candidates, and generate targeted resume versions.</div>",
        unsafe_allow_html=True,
    )

    if "profiles" not in st.session_state:
        st.session_state.profiles = []
    if "job_description" not in st.session_state:
        st.session_state.job_description = load_sample_text(SAMPLE_JD_PATH)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Resume Upload",
            "Job Description",
            "Match Dashboard",
            "Resume Generator",
            "Analytics",
        ],
    )

    if page == "Resume Upload":
        st.header("Resume Upload and Parsing")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        use_sample = st.checkbox("Load sample resume", value=not st.session_state.profiles)

        if st.button("Parse Resumes", type="primary"):
            profiles = []
            if use_sample:
                sample_text = load_sample_text(SAMPLE_RESUME_PATH)
                profiles.append(parse_resume_text(sample_text, "sample_resume.txt"))
            for file in uploaded_files:
                try:
                    profiles.append(parse_uploaded_resume(file))
                except Exception as exc:
                    st.error(f"Could not parse {file.name}: {exc}")
            st.session_state.profiles = profiles

        if st.session_state.profiles:
            parsed_json = json.dumps([profile.to_dict() for profile in st.session_state.profiles], indent=2)
            st.download_button(
                "Download Parsed Resume JSON",
                data=parsed_json,
                file_name="parsed_resumes.json",
                mime="application/json",
            )
            for profile in st.session_state.profiles:
                with st.expander(f"{profile.name} - {profile.file_name}", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.write({"email": profile.email, "phone": profile.phone})
                    col2.write({"skills": profile.skills})
                    st.write("Education", profile.education)
                    st.write("Experience", profile.experience)
                    st.write("Projects", profile.projects)
                    st.write("Certifications", profile.certifications)

    elif page == "Job Description":
        st.header("Job Description Analysis")
        jd_file = st.file_uploader("Upload JD as PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        if jd_file is not None:
            try:
                st.session_state.job_description = extract_text_from_file(jd_file)
            except Exception as exc:
                st.error(f"Could not parse job description file: {exc}")

        st.session_state.job_description = st.text_area(
            "Paste or edit job description",
            value=st.session_state.job_description,
            height=260,
        )

        analysis = analyze_job_description(st.session_state.job_description)
        col1, col2, col3 = st.columns(3)
        col1.metric("Role Category", analysis["role_category"])
        col2.metric("Experience Level", analysis["experience_level"])
        col3.metric("Required Skills Found", len(analysis["required_skills"]))
        st.write("Required skills", analysis["required_skills"])
        st.write("Top keywords", analysis["keywords"])

    elif page == "Match Dashboard":
        st.header("Candidate Ranking Dashboard")
        use_embeddings = st.checkbox("Use BERT sentence embeddings if available", value=False)
        if not st.session_state.profiles:
            st.info("Upload or load at least one resume first.")
            return
        if not st.session_state.job_description.strip():
            st.info("Add a job description first.")
            return

        ranked_df = rank_candidates(
            st.session_state.profiles,
            st.session_state.job_description,
            use_embeddings=use_embeddings,
        )
        save_rankings(ranked_df)
        search = st.text_input("Search candidate")
        min_score = st.slider("Minimum ATS score", 0, 100, 0)

        filtered_df = ranked_df[ranked_df["ats_score"] >= min_score]
        if search:
            filtered_df = filtered_df[filtered_df["candidate"].str.contains(search, case=False, na=False)]

        display_df = filtered_df.copy()
        for column in ["matched_skills", "missing_skills", "missing_keywords", "required_skills", "job_keywords"]:
            display_df[column] = display_df[column].apply(lambda items: ", ".join(items))
        st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "Download Ranking CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="candidate_rankings.csv",
            mime="text/csv",
        )

        if not ranked_df.empty:
            best = ranked_df.iloc[0]
            st.subheader("Top Candidate")
            c1, c2, c3 = st.columns(3)
            c1.metric("Candidate", best["candidate"])
            c2.metric("ATS Score", f"{best['ats_score']:.2f}%")
            c3.metric("Skill Match", f"{best['skill_match_percentage']:.2f}%")
            st.write("Missing skills", best["missing_skills"])
            st.write("Recommended learning path", recommend_skills(best["missing_skills"], best["role_category"]))

    elif page == "Resume Generator":
        st.header("Personalized Resume Generator")
        if not st.session_state.profiles:
            st.info("Upload or load a resume first.")
            return

        names = [profile.name for profile in st.session_state.profiles]
        selected_name = st.selectbox("Choose candidate", names)
        template = st.selectbox("Choose template", ["Modern", "Classic", "Compact"])
        selected_profile = st.session_state.profiles[names.index(selected_name)]
        match = calculate_match(selected_profile, st.session_state.job_description)
        generated = generate_resume(selected_profile, st.session_state.job_description, match, template)

        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            edited_resume = st.text_area("Editable optimized resume", generated, height=620)
            st.download_button(
                "Download Markdown Resume",
                data=edited_resume,
                file_name=f"{selected_profile.name.replace(' ', '_')}_optimized_resume.md",
                mime="text/markdown",
            )
            try:
                pdf_bytes = markdown_to_pdf(edited_resume)
                st.download_button(
                    "Download PDF Resume",
                    data=pdf_bytes,
                    file_name=f"{selected_profile.name.replace(' ', '_')}_optimized_resume.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                st.info(f"PDF export is unavailable until reportlab is installed. Details: {exc}")
        with col2:
            render_score("ATS Score", match["ats_score"])
            render_score("Skill Match", match["skill_match_percentage"])
            st.write("Matched skills", match["matched_skills"])
            st.write("Missing skills", match["missing_skills"])
            st.subheader("Interview Questions")
            for question in generate_interview_questions(match):
                st.write(f"- {question}")

    elif page == "Analytics":
        st.header("Hiring Analytics")
        if not st.session_state.profiles:
            st.info("Upload or load resumes first.")
            return
        ranked_df = rank_candidates(st.session_state.profiles, st.session_state.job_description)

        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates", len(ranked_df))
        c2.metric("Average ATS Score", f"{ranked_df['ats_score'].mean():.2f}%")
        c3.metric("Average Skill Match", f"{ranked_df['skill_match_percentage'].mean():.2f}%")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=ranked_df, x="ats_score", y="candidate", ax=ax, color="#27ae60")
        ax.set_xlabel("ATS Score")
        ax.set_ylabel("Candidate")
        ax.set_title("Candidate Comparison")
        st.pyplot(fig)

        plot_skill_frequency(ranked_df)


if __name__ == "__main__":
    main()
