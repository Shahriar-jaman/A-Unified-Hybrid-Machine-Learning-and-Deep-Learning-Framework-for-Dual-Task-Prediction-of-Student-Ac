from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask.json.provider import DefaultJSONProvider
import mysql.connector
import joblib
import numpy as np
import pandas as pd
import os
from datetime import timedelta
import re
from groq import Groq
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from datetime import datetime

# Optional TensorFlow
try:
    from tensorflow.keras.models import load_model
    print("✅ TensorFlow/Keras available")
except ImportError:
    load_model = None
    print("⚠️ TensorFlow not available - DL models disabled")

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are AIMHESS AI, a professional and empathetic academic mentor for university students.
Your role is to provide structured, actionable, and personalized guidance based on the student's:
- Performance Level (Low / Medium / High)
- Stress Level (Low / Medium / High)
Guidelines:
- Reference both Performance Level and Stress Level in your responses.
- For Low Performance + High Stress: Focus on building foundational habits, stress reduction, and small wins.
- For High Performance + Low Stress: Emphasize optimization, advanced strategies, and sustainability.
- For mixed levels: Balance academic improvement with wellbeing support.
- Always ask about their current feelings and progress from previous suggestions.
- Provide 4-5 clear, realistic, and prioritized action steps.
- Maintain a professional yet warm and encouraging tone.
- Keep responses concise and under 220 words.
- End every response with: "I'm always here to support you."
"""

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAME_SITE'] = 'Lax'

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json = CustomJSONProvider(app)

# Database Setup
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'aimhes_user'),
    'password': os.getenv('DB_PASS', 'aimhes123'),
    'database': os.getenv('DB_NAME', 'aimhes_db'),
    'autocommit': True,
    'raise_on_warnings': True
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# --- Load Models ---
print("Loading models...")
try:
    lgb_acad = joblib.load('saved_models/lgb_acad.pkl')
    lgb_mh = joblib.load('saved_models/lgb_mh.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Models not found, using rule-based fallback:", e)
    lgb_acad = None
    lgb_mh = None

# --- Strong Rule-Based Fallback ---
def rule_based_prediction(form_data):
    try:
        cgpa = float(form_data.get('CGPA', 3.0))
        study_h = float(form_data.get('StudyHoursDaily', 3.0))
        sleep_h = float(form_data.get('SleepHours', 6.5))
        stress = float(form_data.get('AcademicStress', 5.0))
        att_str = str(form_data.get('AttendanceRate', '85')).replace('%', '').strip()
        att = float(att_str) / 100.0 if any(c.isdigit() for c in att_str) else 0.85
        
        # Strong heuristic
        acad_score = cgpa * 25 + study_h * 8 + att * 30 + sleep_h * 4 - stress * 3
        mh_score = stress * 9 + max(0, 8 - sleep_h) * 8 + max(0, 4 - study_h) * 5
        
        acad_score = max(25, min(95, acad_score))
        mh_score = max(15, min(88, mh_score))
        return acad_score, mh_score
    except:
        return 55.0, 52.0

# Keep all preprocessing functions (unchanged)
def preprocess_form_for_ridge(form_data):
    import numpy as np
    def safe_float(val):
        if val is None: return 0.0
        try:
            arr = np.asarray(val).ravel()
            if arr.size > 0:
                return float(arr)
            return 0.0
        except:
            try: return float(val)
            except: return 0.0

    age = safe_float(form_data.get('Age', 20))
    gender = form_data.get('Gender', '0')
    gender_enc = 1.0 if gender == '1' else 0.0
    att_str = form_data.get('AttendanceRate', '85')
    try:
        attendance_rate = safe_float(str(att_str).replace('%', '').strip())
        if attendance_rate > 1.0: attendance_rate /= 100.0
    except:
        attendance_rate = 0.85
    daily_study_hours = safe_float(form_data.get('StudyHoursDaily', 3.0))
    sleep_hours = safe_float(form_data.get('SleepHours', 6.5))
    stress_level = safe_float(form_data.get('AcademicStress', 5.0))
    parental_education_level = safe_float(form_data.get('ParentalEducation', 4.0))
    family_income = safe_float(form_data.get('FamilyIncome', 50000.0))
    motivation_score = safe_float(form_data.get('MotivationScore', 65.0))
    tutoring = form_data.get('PrivateTutoring', 'No')
    private_tutoring = 1.0 if tutoring in ['Yes', '1', True] else 0.0
    internet_quality = safe_float(form_data.get('InternetQuality', 4.0))
    income_log = safe_float(np.log1p(family_income))
    study_sleep_ratio = safe_float(daily_study_hours / (sleep_hours + 1e-5))
    sleep_deficit = safe_float(max(0.0, 8.0 - sleep_hours))
    academic_load = safe_float(daily_study_hours * parental_education_level)
    study_attendance = safe_float(daily_study_hours * attendance_rate)
    income_study = safe_float(income_log * daily_study_hours)
    internet_study = safe_float(internet_quality * daily_study_hours)
    tutoring_motivation = safe_float(private_tutoring * motivation_score)
    wellbeing_acad = safe_float(sleep_hours / (stress_level + 1e-5))
    cgpa = safe_float(form_data.get('CGPA', 3.0))
    pass_fail_enc = 1.0 if cgpa >= 2.0 else 0.0

    features_21 = [
        age, parental_education_level, family_income, income_log,
        daily_study_hours, attendance_rate, sleep_hours,
        stress_level, motivation_score, private_tutoring, internet_quality, gender_enc,
        study_sleep_ratio, sleep_deficit, academic_load, study_attendance,
        income_study, internet_study, tutoring_motivation, wellbeing_acad, pass_fail_enc
    ]
    return np.array([features_21], dtype=np.float32)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = generate_password_hash(request.form['password'])
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            flash("Account created! Please log in.", "success")
            return redirect('/login')
        except mysql.connector.IntegrityError:
            flash("Username already taken.", "danger")
        finally:
            cursor.close()
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash("Welcome back!", "success")
            return redirect(url_for('home'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('home.html', username=session['username'])

@app.route('/risk', methods=['GET', 'POST'])
def risk():
    if 'user_id' not in session:
        return redirect('/login')
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            
            if lgb_acad is None or lgb_mh is None:
                pred_acad_score, pred_mh_score = rule_based_prediction(form_data)
            else:
                try:
                    X = preprocess_form_for_ridge(form_data)
                    raw_acad = lgb_acad.predict(X)[0]
                    raw_mh = lgb_mh.predict(X)[0]
                    pred_acad_score = float(raw_acad)
                    pred_mh_score = float(raw_mh)
                    if pred_acad_score <= 1.0: pred_acad_score *= 100
                    if pred_mh_score <= 1.0: pred_mh_score *= 100
                except:
                    pred_acad_score, pred_mh_score = rule_based_prediction(form_data)
            
            pred_acad_score = max(20, min(95, pred_acad_score))
            pred_mh_score = max(15, min(88, pred_mh_score))
            
            session['perf_level'] = "High" if pred_acad_score >= 75 else "Moderate" if pred_acad_score >= 45 else "Low"
            session['perf_level_score'] = f"{round(pred_acad_score, 1)}/100"
            session['stress_level'] = "High" if pred_mh_score >= 60 else "Moderate" if pred_mh_score >= 30 else "Low"
            session['stress_level_score'] = f"{round(pred_mh_score, 1)}/100"
            
            overall_risk = round(((100 - pred_acad_score) + pred_mh_score) / 2, 1)
            session['student_risk_score'] = overall_risk
            session['student_risk_level'] = "High" if overall_risk >= 65 else "Medium" if overall_risk >= 35 else "Low"
            
            session['acad_predictions'] = {}
            session['mh_predictions'] = {}
            session.modified = True
            
            flash("Assessment completed successfully!", "success")
            return redirect(url_for('result'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Error: {str(e)}", "danger")
            return redirect('/risk')
    
    return render_template('risk_questionnaire.html')

@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect('/login')
    if 'student_risk_score' not in session:
        flash("No active assessment data found. Please complete the diagnostic first.", "warning")
        return redirect('/risk')
    
    p_score = float(str(session.get('perf_level_score', '0/100')).replace('/100', ''))
    s_score = float(str(session.get('stress_level_score', '0/100')).replace('/100', ''))
    
    dl_acad_corrected = round(p_score, 1)
    dl_mh_corrected = round(s_score, 1)
    
    original_risk = session.get('student_risk_score', 0.0)
    corrected_risk = round(((100.0 - dl_acad_corrected) + dl_mh_corrected) / 2.0, 1)
    corrected_level = "High" if corrected_risk >= 65 else "Medium" if corrected_risk >= 35 else "Low"
    
    return render_template(
        'result.html',
        p_score=p_score,
        dl_acad_corrected=dl_acad_corrected,
        s_score=s_score,
        dl_mh_corrected=dl_mh_corrected,
        original_risk=original_risk,
        corrected_risk=corrected_risk,
        corrected_level=corrected_level
    )

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'user_id' not in session:
        return redirect('/login')
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        msg = request.form.get('msg', '').strip()
        if msg:
            session['chat_history'].append({"role": "user", "content": msg})
            
            # === Fetch Prediction Values ===
            perf_level = session.get('perf_level', 'Moderate')
            stress_level = session.get('stress_level', 'Moderate')
            perf_score = session.get('perf_level_score', 'N/A')
            stress_score = session.get('stress_level_score', 'N/A')
            overall_risk = session.get('student_risk_score', 'N/A')
            risk_level = session.get('student_risk_level', 'Unknown')
            
            last_plan = session.get('last_plan', 'No previous suggestions yet')
            
            # Rich context sent to AI
            context = f"""
Student Profile Summary:
- Academic Performance: {perf_level} ({perf_score})
- Mental Health Stress: {stress_level} ({stress_score})
- Overall Risk Level: {risk_level} ({overall_risk}%)
- Previous Suggestions: {last_plan}
"""

            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{context}\n\nStudent's message: {msg}"}
                    ],
                    temperature=0.72,
                    max_tokens=650
                )
                reply = completion.choices[0].message.content.strip()
                
                # === Improved Plan Extraction ===
                plan_match = re.search(
                    r'(?:action steps?|recommendations?|suggested plan|here.?s what|prioriti[sz]ed steps?|suggestions).*?'
                    r'(?=I\'m always here to support you\.|Best regards|Take care|You got this|\Z)',
                    reply, 
                    re.I | re.S | re.M
                )
                
                if plan_match:
                    extracted = plan_match.group(0).strip()
                    # Clean multiple newlines
                    extracted = re.sub(r'\n\s*\n', '\n\n', extracted)
                    session['last_plan'] = extracted[:600]
                else:
                    # Fallback: take last portion of reply
                    session['last_plan'] = reply[-450:].strip()
                
            except Exception as e:
                print("Groq API Error:", str(e))
                reply = "I'm right here to support you. Could you share more about how you're feeling or what challenge you're facing right now?"
                session['last_plan'] = "No previous plan"
            
            session['chat_history'].append({"role": "ai", "content": reply})
            session.modified = True
    
    return render_template('chatbot.html', chat_history=session.get('chat_history', []))


# ====================== AI POWERED DAILY PLANNER ======================
@app.route('/planner')
def planner():
    if 'user_id' not in session:
        return redirect('/login')
    
    if 'perf_level' not in session:
        flash("Please complete the Risk Assessment first!", "warning")
        return redirect('/risk')
    
    perf_level = session.get('perf_level', 'Moderate')
    stress_level = session.get('stress_level', 'Moderate')
    perf_score = session.get('perf_level_score', 'N/A')
    stress_score = session.get('stress_level_score', 'N/A')
    risk_level = session.get('student_risk_level', 'Medium')

    # Generate planner using AI
    planner_data = generate_ai_daily_planner(perf_level, stress_level, perf_score, stress_score, risk_level)
    
    return render_template('planner.html', 
                         planner=planner_data,
                         perf_level=perf_level,
                         stress_level=stress_level)

def generate_ai_daily_planner(perf_level, stress_level, perf_score, stress_score, risk_level):
    """Generate personalized daily planner using Groq AI"""
    prompt = f"""
Create a realistic, detailed daily planner for a university student.

Student Profile:
- Academic Performance: {perf_level} ({perf_score})
- Stress Level: {stress_level}
- Overall Risk: {risk_level}

Requirements:
- Make a hourly schedule from 6:00 AM to 11:00 PM
- Balance study, rest, exercise, meals, and relaxation
- Adjust intensity based on performance and stress level
- Include short breaks, mindfulness, and sleep recommendation
- Use warm, encouraging tone
- Format output as clear time → task

Return only the schedule in this format:
Date: [Today]
Schedule:
06:30 - 07:00 → Task here
...
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert academic life coach. Create practical daily plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        ai_response = completion.choices[0].message.content.strip()
        
        # Parse into structured data
        planner = {
            "date": datetime.now().strftime("%A, %B %d, %Y"),
            "perf_level": perf_level,
            "stress_level": stress_level,
            "perf_score": perf_score,
            "stress_score": stress_score,
            "ai_plan": ai_response
        }
        return planner
        
    except Exception as e:
        print("AI Planner Error:", e)
        # Fallback
        return {
            "date": datetime.now().strftime("%A, %B %d, %Y"),
            "perf_level": perf_level,
            "stress_level": stress_level,
            "perf_score": perf_score,
            "stress_score": stress_score,
            "ai_plan": "AI couldn't generate planner right now.\n\nPlease try again or contact support."
        }
@app.route('/planner/pdf')
def planner_pdf():
    if 'user_id' not in session or 'perf_level' not in session:
        flash("No planner available", "danger")
        return redirect('/planner')
    
    planner = generate_ai_daily_planner(
        session.get('perf_level'),
        session.get('stress_level'),
        session.get('perf_level_score'),
        session.get('stress_level_score'),
        session.get('student_risk_level', 'Medium')
    )
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 80
    
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, y, "AIMHESS AI Daily Planner")
    y -= 40
    
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Date: {planner['date']}")
    y -= 25
    c.drawString(50, y, f"Performance: {planner['perf_level']} | Stress: {planner['stress_level']}")
    y -= 40
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "YOUR DAILY PLAN")
    y -= 30
    
    c.setFont("Helvetica", 11)
    lines = planner['ai_plan'].split('\n')
    for line in lines:
        if y < 50:
            c.showPage()
            y = height - 80
        c.drawString(50, y, line[:90])  # limit line length
        y -= 18
    
    c.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, 
                    download_name=f"AIMHESS_Planner_{datetime.now().strftime('%Y-%m-%d')}.pdf", 
                    mimetype='application/pdf')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)