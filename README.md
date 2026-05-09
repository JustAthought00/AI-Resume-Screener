# AI Resume Screening, Job Matching, and Personalized Resume Generator

This is a Python, NLP, and machine learning project for smart recruitment workflows. It parses resumes, analyzes job descriptions, calculates ATS/job match scores, ranks candidates, recommends missing skills, and generates personalized optimized resumes for specific roles.

## Core Features

- Upload PDF, DOCX, or TXT resumes
- Extract name, email, phone, skills, education, experience, projects, and certifications
- Analyze job descriptions from text input or file upload
- Extract required skills, keywords, experience level, and role category
- Match resumes to jobs using TF-IDF and cosine similarity
- Optional BERT-style sentence embeddings with `sentence-transformers`
- Calculate ATS score, skill match percentage, missing skills, and missing keywords
- Rank multiple candidates for HR screening
- Save latest ranking output with Joblib for review or audit
- Generate company/job-specific resume versions without inventing fake information
- Export generated resume as Markdown or PDF
- Show analytics charts for candidate comparison and skill frequency
- Generate interview questions and skill recommendations

## Project Structure

```text
.
|-- app.py
|-- resume_parser.py
|-- job_matcher.py
|-- resume_generator.py
|-- preprocessing.py
|-- requirements.txt
|-- README.md
|-- models/
|   `-- .gitkeep
|-- datasets/
|   |-- sample_resume.txt
|   `-- sample_job_description.txt
|-- templates/
|   |-- modern_resume.md
|   `-- classic_resume.md
`-- static/
    `-- styles.css
```

## How the System Works

### 1. Resume Parsing

`resume_parser.py` reads resumes from PDF, DOCX, or TXT files. It extracts text, then uses regular expressions, spaCy, and section parsing to identify:

- Candidate name
- Email
- Phone
- Skills
- Education
- Experience
- Projects
- Certifications

The extracted result is stored as a structured `ParsedProfile` object.

### 2. Job Description Analysis

The job description analyzer extracts:

- Required skills using a curated technical skill library
- High-frequency keywords
- Experience level such as Entry-level, Mid-level, or Senior
- Role category such as Data Science, Frontend Development, Backend Development, or Cloud/DevOps

### 3. Matching Engine

`job_matcher.py` compares a resume with the job description using:

- NLP preprocessing
- TF-IDF vectorization
- Cosine similarity
- Skill overlap
- Optional sentence-transformer embeddings

The ATS score is calculated from text similarity, skill match, and semantic similarity.

### 4. Personalized Resume Generator

`resume_generator.py` creates a truthful, job-specific resume by:

- Reordering skills based on relevance to the job description
- Highlighting relevant projects and experience first
- Creating a targeted summary
- Adding optimization suggestions
- Listing missing skills without pretending the candidate has them
- Exporting the resume as editable Markdown or PDF

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Install the recommended spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Run the app:

```bash
streamlit run app.py
```

## Usage Steps

1. Open the Streamlit app.
2. Go to `Resume Upload`.
3. Upload PDF/DOCX resumes or load the included sample resume.
4. Go to `Job Description`.
5. Paste a JD or use the included sample JD.
6. Open `Match Dashboard` to view ranked candidates and missing skills.
7. Open `Resume Generator` to generate a tailored resume.
8. Download the optimized resume as Markdown or PDF.
9. Open `Analytics` to view candidate comparison charts.

## Sample Dataset Links

You can test or extend the project with public resume and job datasets:

- Kaggle Resume Dataset: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- Resume Dataset by Category: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- Job Description Dataset: https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
- LinkedIn Job Postings Dataset: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

## Streamlit Cloud Deployment

1. Push this project to GitHub.
2. Go to https://streamlit.io/cloud.
3. Create a new app.
4. Select your repository and branch.
5. Set the main file path to `app.py`.
6. Add `requirements.txt`.
7. Deploy.

For best spaCy support on Streamlit Cloud, add this command to package setup documentation or use a prebuilt model package if needed:

```bash
python -m spacy download en_core_web_sm
```

The app still works with a blank spaCy fallback if the small English model is not installed, but name extraction will be more accurate with the model.

## Notes for GitHub Portfolio

- The included sample files make the app demo-ready.
- The code is modular and beginner-friendly.
- The generated resume avoids fake claims and only reorganizes or suggests improvements based on provided candidate data.
- For production use, expand the skill library, add database storage, include authentication, and validate extracted data with human review.
