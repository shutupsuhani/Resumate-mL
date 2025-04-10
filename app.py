from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
import io

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
You are an advanced and Expert Applicant Tracking Score Software (ATS). Your task is to compare the uploaded resume against the provided job description.

Evaluate the following:
1. Keyword match (skills, qualifications, job titles, technologies).
2. Educational alignment.
3. Relevant experience match.
4. Certifications and achievements.
5. Overall alignment with the job role.

Based on the above criteria, calculate an **overall ATS Match Score** between 0 and 100 percent.

Be accurate and strict, and show only:
- The final ATS Match Score (out of 100) — nothing else.

Also point the positive points and negative points in points.
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

def default_statement():
    return "Welcome to Resumate Backend"
# Endpoints

@app.route("/",methods=["GET"])
def local_route():
    return default_statement()

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

        # Read and extract resume text
        resume_text = extract_text_from_pdf(file.stream)

        # Get Gemini response
        result = get_gemini_response(job_description, resume_text, prompt)
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
