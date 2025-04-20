import os
import logging
import socket
from flask import Flask, render_template, request, session, jsonify, redirect
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import re

load_dotenv()

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': os.path.abspath('uploads'),
    'ALLOWED_EXTENSIONS': {'pdf'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'SECRET_KEY': os.urandom(24)
})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MEDICAL_KEYWORDS = {
    'blood', 'cholesterol', 'urine', 'report', 'test', 'doctor', 'symptom', 'diagnosis', 'treatment',
    'creatinine', 'wbc', 'neutrophils', 'lymphocytes', 'vldl', 'hba1c', 'rbc', 'platelet', 'cbc',
    'hdl', 'ldl', 'triglycerides', 'glucose', 'bilirubin', 'protein', 'bun', 'sodium', 'potassium',
    'chlorides', 'uric acid', 'phosphorous', 'calcium', 'mcv', 'mch', 'mchc', 'rdw', 'esr', 'thyroid',
    'vitamin', 'infection', 'urinalysis', 'urine culture', 'liver', 'kidney', 'hematology', 'cbc', 'lipid'
}

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
MODEL = genai.GenerativeModel('gemini-1.5-pro')


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def is_medical_query(query):
    return any(keyword in query.lower() for keyword in MEDICAL_KEYWORDS)


def process_pdf(filepath):
    try:
        with open(filepath, 'rb') as f:
            return '\n'.join(
                page.extract_text()
                for page in PyPDF2.PdfReader(f).pages
                if page.extract_text()
            )
    except Exception as e:
        logging.error(f"PDF processing failed: {str(e)}")
        return None


def parse_results(analysis):
    # Parse bullet points and classify them
    results = []
    lines = re.findall(r"[-*•] (.+)", analysis)

    for line in lines:
        text = line.strip()
        lowered = text.lower()

        if any(w in lowered for w in ['high', 'elevated', 'increased', 'above']):
            status = 'high'
        elif any(w in lowered for w in ['low', 'decreased', 'reduced', 'below']):
            status = 'low'
        elif any(w in lowered for w in ['normal', 'within range', 'okay']):
            status = 'normal'
        else:
            status = 'normal'  # Default fallback

        results.append({
            'label': text,
            'status': status
        })

    return results


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if not (file and allowed_file(file.filename)):
            return render_template('index.html', error="Invalid file. Please upload a PDF.")

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            report_text = process_pdf(filepath)
            if not report_text:
                raise ValueError("Could not extract text from PDF")

            # Ask Gemini to analyze
            analysis = MODEL.generate_content(
                f"""You are a medical expert. Analyze the following report and provide:
- 🚨 Abnormal Findings (bullet points)
- ✅ Normal Results (bullet points)
- 💡 Simple Explanations
- 📋 Recommendations (1-2 lines each)

Format in clear markdown. Be concise and helpful.

Medical Report:
{report_text}
"""
            ).text

            session['analysis'] = analysis
            structured_results = parse_results(analysis)

            return render_template('results.html', results=structured_results)

        except Exception as e:
            logging.error(str(e))
            return render_template('index.html', error="Error processing file. Please try again.")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_question = data.get('question', '').strip()

        if not is_medical_query(user_question):
            return jsonify({
                'response': "I can only answer questions about your medical report or related health results. Please ask about your test results, findings, or medical terms."
            })

        prompt = (
            f"You are a medical report assistant. Only answer questions about the user's uploaded medical report or related health results. "
            f"If the question is unrelated, politely reply: 'I can only answer questions about your medical report or related health results.'\n\n"
            f"User's question: {user_question}\n\n"
            f"Medical Report Analysis:\n{session.get('analysis', '')}\n"
            f"Keep your answer under 3 sentences, clear and professional."
        )

        response = MODEL.generate_content(prompt)
        return jsonify({'response': response.text})

    except Exception as e:
        logging.error(f"Chatbot error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = find_free_port()
    logging.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
