from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
import io
import re

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Prompts
PROMPT_ANALYZE = """
You are an experienced Technical Human Resource Manager. Your task is to analyze the uploaded resume in the context of the provided job description. 
Evaluate whether the candidate’s profile aligns with the job role. List strengths, weaknesses, and give a short hiring recommendation.
"""

PROMPT_SKILLS = """
You are a skill gap analyzer. Based on the resume and the job description, identify the missing or weak skills and suggest ways to improve.
"""

PROMPT_MATCH_SCORE = """
You are an advanced Applicant Tracking System (ATS). Your task is to analyze the candidate’s resume against the job description.

Evaluate the following:
1. Keyword match (skills, tools, titles)
2. Education and qualifications
3. Relevant experience
4. Certifications or achievements

Then, provide:

1. An **ATS Match Score** between 0 and 100.
2. **Positive points** (at least 3)
3. **Negative points** (at least 3)

Follow this exact output format:
---
ATS Match Score: 87%

Positive Points:
- Strong experience with React.js
- Relevant degree in Computer Science
- Used many keywords from job description

Negative Points:
- Lacks required AWS certification
- Only 1 year of work experience
- Missing some soft skills

---
Be accurate and use the exact format. Don't add any extra explanation.
"""

# Helper to extract PDF text
def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Gemini call
def get_gemini_response(input_text, resume_text, prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    content = [input_text, resume_text, prompt]
    response = model.generate_content(content)
    return response.text

# Parse Gemini output for ATS score
def parse_gemini_output(text):
    # Extract score
    score_match = re.search(r'ATS Match Score\s*[:\-]?\s*(\d{1,3})\s*%', text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None

    # Extract positive points
    positive_match = re.search(r'Positive Points:\s*((?:- .+\n?)+)', text)
    positives = positive_match.group(1).strip().split('\n') if positive_match else []

    # Extract negative points
    negative_match = re.search(r'Negative Points:\s*((?:- .+\n?)+)', text)
    negatives = negative_match.group(1).strip().split('\n') if negative_match else []

    # Clean the lines
    positives = [p.lstrip("- ").strip() for p in positives]
    negatives = [n.lstrip("- ").strip() for n in negatives]

    return {
        "score": score,
        "positives": positives,
        "negatives": negatives,
        "type": "match"
    }

# Routes
@app.route("/", methods=["GET"])
def home():
    return "Welcome to Resumate Backend!"

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    return handle_request(PROMPT_ANALYZE)

@app.route("/skill_gap", methods=["POST"])
def skill_gap():
    return handle_request(PROMPT_SKILLS)

@app.route("/match_score", methods=["POST"])
def match_score():
    return handle_request(PROMPT_MATCH_SCORE)

# Core handler
def handle_request(prompt):
    try:
        job_description = request.form.get("job_description")
        file = request.files.get("resume")

        if not job_description or not file:
            return jsonify({"error": "Missing job description or resume file"}), 400

        resume_text = extract_text_from_pdf(file.stream)
        result = get_gemini_response(job_description, resume_text, prompt)

        if prompt == PROMPT_MATCH_SCORE:
            return jsonify(parse_gemini_output(result))
        else:
            return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
