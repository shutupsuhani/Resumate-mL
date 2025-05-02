from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
import re
import json

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
You are an expert in resume screening and applicant tracking systems (ATS).

Given a **job description** and a **resume**, perform the following tasks and return your response in **JSON format** like this:

{
  "ATS_Match_Score": <score out of 100>,
  "Matching_Skills": [list of skills found in both],
  "Missing_Skills": [list of skills found only in job description],
  "Recommendations": [list of actionable recommendations to improve the score]
}

### Job Description:
{job_description}

### Resume:
{resume_text}
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
def get_gemini_response(job_description, resume_text, prompt_template):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    formatted_prompt = prompt_template.format(
        job_description=job_description,
        resume_text=resume_text
    )
    response = model.generate_content(formatted_prompt)
    return response.text

# Parse Gemini structured JSON response
def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse structured output from Gemini."}

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
    return handle_request(PROMPT_MATCH_SCORE, expect_json=True)

@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Missing resume file or job description'}), 400

    resume_file = request.files['resume']
    job_description = request.form.get('job_description')

    resume_text = extract_text_from_pdf(resume_file)

    prompt = f"""
    Generate a tailored, professional cover letter using the following resume and job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    The cover letter should be concise (one page), highlight the candidate's most relevant skills and experiences,
    and be ATS-friendly.
    """

    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content(prompt)
    cover_letter = response.text.strip()

    return jsonify({'cover_letter': cover_letter})

# Core handler
def handle_request(prompt, expect_json=False):
    try:
        job_description = request.form.get("job_description")
        file = request.files.get("resume")

        if not job_description or not file:
            return jsonify({"error": "Missing job description or resume file"}), 400

        resume_text = extract_text_from_pdf(file.stream)
        result = get_gemini_response(job_description, resume_text, prompt)

        if expect_json:
            return jsonify(parse_json_response(result))
        else:
            return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
