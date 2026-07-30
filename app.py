from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import io
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = 'healthbridge_final'

# === NATURAL LANGUAGE CHATBOT ===
symptom_synonyms = {
    # --- General Symptoms ---
    'fever': ['fever', 'temperature', 'high temp', 'hot', 'feverish'],
    'cough': ['cough', 'coughing', 'hacking'],
    'sneezing': ['sneeze', 'sneezing'],
    'runny_nose': ['runny nose', 'watery nose'],
    'sore_throat': ['sore throat', 'throat pain'],
    'body_ache': ['body ache', 'body pain', 'muscle pain'],
    'itchy_eyes': ['itchy eyes', 'eyes itchy'],
    'rash': ['rash', 'skin rash'],
    'headache': ['headache', 'head pain'],
    'fatigue': ['tired', 'fatigue', 'exhausted', 'weakness'],
    
    # --- Diabetes Symptoms ---
    'thirst': ['thirsty', 'thirst', 'dry mouth', 'drinking water'],
    'freq_urination': ['frequent urination', 'peeing a lot', 'bathroom often', 'urine'],
    'hunger': ['hungry', 'hunger', 'eating a lot'],
    'weight_loss': ['weight loss', 'losing weight', 'thin'],

    # --- Heart Symptoms ---
    'chest_pain': ['chest pain', 'chest pressure', 'chest tight', 'heart pain'],
    'breathlessness': ['breathless', 'short of breath', 'breathing problem', 'panting', 'hard to breathe'],
    'palpitations': ['heart beat', 'racing heart', 'palpitations', 'fluttering'],

    # --- Kidney Symptoms ---
    'swelling': ['swelling', 'swollen', 'puffiness', 'edema', 'swollen legs', 'swollen ankles'],
    'back_pain': ['back pain', 'lower back pain', 'flank pain'],
    
    # --- Liver Symptoms ---
    'jaundice': ['yellow skin', 'yellow eyes', 'jaundice', 'yellowish'],
    'nausea': ['nausea', 'vomit', 'vomiting', 'puking', 'feeling sick'],
    'abdominal_pain': ['stomach pain', 'abdominal pain', 'belly pain'],

    #----- stroke symptoms ---

    'face_drop': ['face', 'drooping', 'numb face', 'smile looks wrong', 'facial', 'mouth droop'],
    'arm_weakness': ['arm', 'weakness', 'cant lift arm', 'numb arm', 'leg weakness', 'paralysis'],
    'speech_difficulty': ['speech', 'slurred', 'cant speak', 'confused speech', 'talking funny'],
    'confusion': ['confusion', 'confused', 'dont understand', 'disoriented'],
    'vision_problems': ['vision', 'blurred', 'cant see', 'blind', 'double vision'],
    'severe_headache': ['severe headache', 'worst headache', 'thunderclap headache']
}

simple_diseases = {
    # --- Simple Diseases ---
    "Viral Fever": ["fever", "body_ache", "fatigue"],
    "Common Cold": ["cough", "sneezing", "runny_nose", "sore_throat"],
    "Allergy": ["itchy_eyes", "sneezing", "rash"],
    "Headache": ["headache", "fatigue"],
    "Persistent Cough": ["cough", "fatigue", "fever"],

    # --- Serious Diseases (Clinical Based) ---
    "Diabetes": ["thirst", "freq_urination", "hunger", "fatigue", "weight_loss"],
    "Heart Disease": ["chest_pain", "breathlessness", "fatigue", "palpitations"],
    "Kidney Disease": ["swelling", "fatigue", "freq_urination", "back_pain"],
    "Liver Disease": ["jaundice", "nausea", "fatigue", "abdominal_pain", "swelling"],
    "Stroke": ["face_droop", "arm_weakness", "speech_difficulty", "confusion", "vision_problems", "severe_headache"]
}

def analyze_symptoms_natural(text):
    if not text or not text.strip():
        return "General Wellness", "Low"
    text = text.lower()
    matched_symptoms = []
    for symptom_key, synonyms in symptom_synonyms.items():
        if any(word in text for word in synonyms):
            matched_symptoms.append(symptom_key)
    best_disease = "General Wellness"
    best_score = 0
    for disease, keywords in simple_diseases.items():
        score = len(set(keywords) & set(matched_symptoms))
        if score > best_score:
            best_score = score
            best_disease = disease
    
    risk = "Low"
    if best_score >= 3: risk = "Moderate"
    if best_score >= 4: risk = "High"

    serious_list = ["Diabetes", "Heart Disease", "Kidney Disease", "Liver Disease"]
    if best_disease in serious_list and best_score >= 1:
        risk = "High"

    return best_disease, risk

# === DATABASE SETUP ===
def init_db():
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            password TEXT NOT NULL,
            specialization TEXT NOT NULL DEFAULT 'General Physician'
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            disease TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            clinical_data TEXT,
            symptom_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    # UPDATED TABLE: Added prescription column
    c.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            report_id INTEGER NOT NULL,
            message TEXT,
            doctor_suggestion TEXT,
            prescription TEXT,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(doctor_id) REFERENCES doctors(id),
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultation_id INTEGER NOT NULL,
            sender TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(consultation_id) REFERENCES consultations(id)
        )
    ''')
    try:
        c.execute('INSERT INTO doctors (name, email, password, specialization) VALUES (?, ?, ?, ?)',
                  ('Dr. Smith', 'doctor@example.com', generate_password_hash('health123'), 'General Physician'))
    except:
        pass
    conn.commit()
    conn.close()

# === MODEL TRAINING ===
def train_models():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # ------------------ DIABETES ------------------
    if os.path.exists('models/diabetes.pkl'): os.remove('models/diabetes.pkl')
    n_features_diabetes = 5 
    cols = ['Glucose','BMI','Age','Pregnancies','Insulin']
    if os.path.exists('data/diabetes.csv'):
        df = pd.read_csv('data/diabetes.csv', header=None)
        df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        X = df[cols].fillna(df[cols].mean())
        y = df['Outcome']
    else:
        X_mock, y_mock = make_classification(n_samples=768, n_features=n_features_diabetes, n_informative=3, random_state=42)
        df = pd.DataFrame(X_mock, columns=cols)
        df['Glucose'] = np.clip(df['Glucose'] * 30 + 100, 50, 200)
        df['BMI'] = np.clip(df['BMI'] * 10 + 25, 15, 50)
        X = df
        y = (y_mock > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'models/diabetes.pkl')

    # ------------------ HEART ------------------
    if os.path.exists('models/heart.pkl'): os.remove('models/heart.pkl')
    n_features_heart = 6
    cols = ['cp','chol','thalach','exang','oldpeak','age']
    if os.path.exists('data/heart.csv'):
        df = pd.read_csv('data/heart.csv')
        X = df[cols].fillna(df[cols].median())
        y = df['target'] if 'target' in df.columns else (df['chol'] > 240).astype(int)
    else:
        X_mock, y_mock = make_classification(n_samples=303, n_features=n_features_heart, n_informative=4, random_state=1)
        df = pd.DataFrame(X_mock, columns=cols)
        df['chol'] = np.clip(df['chol'] * 50 + 200, 100, 400)
        X = df
        y = (y_mock > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(X, y)
    joblib.dump(model, 'models/heart.pkl')

    # ------------------ KIDNEY ------------------
    if os.path.exists('models/kidney.pkl'): os.remove('models/kidney.pkl')
    n_features_kidney = 5
    cols = ['blood_urea','serum_creatinine','age','hypertension','diabetes']
    if os.path.exists('data/kidney.csv'):
        df = pd.read_csv('data/kidney.csv')
        X = df[cols].fillna(df[cols].median())
        y = df['classification'] if 'classification' in df.columns else (df['serum_creatinine'] > 1.3).astype(int)
    else:
        X_mock, y_mock = make_classification(n_samples=400, n_features=n_features_kidney, n_informative=3, random_state=2)
        df = pd.DataFrame(X_mock, columns=cols)
        df['serum_creatinine'] = np.clip(df['serum_creatinine'] * 0.5 + 1.0, 0.5, 5.0)
        X = df
        y = (y_mock > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X, y)
    joblib.dump(model, 'models/kidney.pkl')

    # ------------------ LIVER ------------------
    if os.path.exists('models/liver.pkl'): os.remove('models/liver.pkl')
    n_features_liver = 5
    cols = ['Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Albumin','Age']
    if os.path.exists('data/liver.csv'):
        df = pd.read_csv('data/liver.csv')
        X = df[cols].fillna(df[cols].median())
        y = df['Dataset'] if 'Dataset' in df.columns else (df['Total_Bilirubin'] > 1.2).astype(int)
    else:
        X_mock, y_mock = make_classification(n_samples=583, n_features=n_features_liver, n_informative=3, random_state=3)
        df = pd.DataFrame(X_mock, columns=cols)
        df['Total_Bilirubin'] = np.clip(df['Total_Bilirubin'] * 1.0 + 0.8, 0.1, 5.0)
        X = df
        y = (y_mock > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=3)
    model.fit(X, y)
    joblib.dump(model, 'models/liver.pkl')

    #-------------------------stroke---------------------------

    if os.path.exists('models/stroke.pkl'): os.remove('models/stroke.pkl')
    
    # Features in your new CSV: age, avg_glucose_level, bmi, hypertension, heart_disease
    stroke_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    
    if os.path.exists('data/stroke.csv'):
        df = pd.read_csv('data/stroke.csv')
        X = df[stroke_features].fillna(df[stroke_features].mean())
        y = df['stroke']
    else:
        # Fallback if CSV is missing (though you now have it!)
        X_mock, y_mock = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=5)
        df = pd.DataFrame(X_mock, columns=stroke_features)
        X = df
        y = (y_mock > 0.5).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=5)
    model.fit(X_scaled, y)
    
    joblib.dump(model, 'models/stroke.pkl')
    joblib.dump(scaler, 'models/stroke_scaler.pkl')

_initialized = False
@app.before_request
def initialize():
    global _initialized
    if not _initialized:
        if os.path.exists('models/diabetes.pkl'): os.remove('models/diabetes.pkl')
        if os.path.exists('models/heart.pkl'): os.remove('models/heart.pkl')
        if os.path.exists('models/kidney.pkl'): os.remove('models/kidney.pkl')
        if os.path.exists('models/liver.pkl'): os.remove('models/liver.pkl')
        if os.path.exists('models/cancer.pkl'): os.remove('models/cancer.pkl')
        if os.path.exists('models/cancer_scaler.pkl'): os.remove('models/cancer_scaler.pkl')
        init_db()
        train_models()
        _initialized = True

# === AUTH ROUTES ===
@app.route('/auth/patient/register', methods=['GET', 'POST'])
def patient_register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect('healthbridge.db')
            c = conn.cursor()
            c.execute('INSERT INTO patients (name, email, password) VALUES (?, ?, ?)', (name, email, password))
            conn.commit()
            conn.close()
            return redirect('/auth/patient/login')
        except sqlite3.IntegrityError:
            return render_template('auth/patient_register.html', error="Email exists")
    return render_template('auth/patient_register.html')

@app.route('/auth/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('healthbridge.db')
        c = conn.cursor()
        c.execute('SELECT id, name, password FROM patients WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['role'] = 'patient'
            return redirect('/patient/dashboard')
        return render_template('auth/patient_login.html', error="Invalid credentials")
    return render_template('auth/patient_login.html')

@app.route('/auth/doctor/register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        specialization = request.form['specialization']
        try:
            conn = sqlite3.connect('healthbridge.db')
            c = conn.cursor()
            c.execute('INSERT INTO doctors (name, email, password, specialization) VALUES (?, ?, ?, ?)',
                      (name, email, password, specialization))
            conn.commit()
            conn.close()
            return redirect('/auth/doctor/login')
        except sqlite3.IntegrityError:
            return render_template('auth/doctor_register.html', error="Email exists")
    return render_template('auth/doctor_register.html')

@app.route('/auth/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('healthbridge.db')
        c = conn.cursor()
        c.execute('SELECT id, name, password FROM doctors WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['role'] = 'doctor'
            return redirect('/doctor/dashboard')
        return render_template('auth/doctor_login.html', error="Invalid credentials")
    return render_template('auth/doctor_login.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient/dashboard')
def patient_dashboard():
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    return render_template('patient/dashboard.html')

@app.route('/patient/check_risk')
def check_risk():
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    return render_template('patient/check_risk.html')

# === PREDICTION ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data'}), 400
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401

        conn = sqlite3.connect('healthbridge.db')
        c = conn.cursor()
        
        # === UPDATED LOGIC: Clinical is now Doctor-Only ===
        if data.get('mode') == 'clinical':
            if session.get('role') != 'doctor':
                return jsonify({'error': 'Access Denied: Only doctors perform clinical assessments.'}), 403
            
            # Doctor must provide patient_id
            patient_id = data.get('patient_id')
            if not patient_id:
                return jsonify({'error': 'Patient ID required.'}), 400

            disease = data.get('disease')
            model_path = f'models/{disease}.pkl'
            scaler_path = f'models/{disease}_scaler.pkl'
            
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model {disease} not ready.'}), 500

            model = joblib.load(model_path)
            
            feature_map = {
                'diabetes': ['Glucose','BMI','Age','Pregnancies','Insulin'],
                'heart': ['cp','chol','thalach','exang','oldpeak','age'],
                'kidney': ['blood_urea','serum_creatinine','age','hypertension','diabetes'],
                'liver': ['Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Albumin','Age'],
                'stroke': ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']  # <--- NEW
            }
            
            features = feature_map[disease]
            inputs = [float(data.get(f, 0)) for f in features]
            input_df = pd.DataFrame([inputs], columns=features)
            
            if disease == 'stroke' and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df
            
            prob = model.predict_proba(input_scaled)[0][1]
            risk = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"
            
            # Insert report using the selected PATIENT_ID
            c.execute('''INSERT INTO reports (patient_id, mode, disease, risk_level, clinical_data)
                         VALUES (?, ?, ?, ?, ?)''',
                      (patient_id, 'clinical', disease, risk, str(inputs)))
            
        else:
            # Chatbot logic (Patient Only)
            if session.get('role') != 'patient':
                return jsonify({'error': 'Only patients use symptom chat.'}), 403
                
            symptoms_text = data.get('symptoms', '')
            disease, risk = analyze_symptoms_natural(symptoms_text)
            c.execute('''INSERT INTO reports (patient_id, mode, disease, risk_level, symptom_text)
                         VALUES (?, ?, ?, ?, ?)''',
                      (session['user_id'], 'chatbot', disease, risk, symptoms_text))
        
        conn.commit()
        report_id = c.lastrowid
        conn.close()
        return jsonify({
            'report_id': report_id,
            'disease': disease,
            'risk': risk,
            'is_high': risk == "High"
        })
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': f'Prediction Failed: {str(e)}'}), 500

@app.route('/patient/result/<int:report_id>')
def patient_result(report_id):
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    report = conn.execute('SELECT * FROM reports WHERE id = ?', (report_id,)).fetchone()
    doctors = conn.execute('SELECT id, name, specialization FROM doctors').fetchall()
    conn.close()
    return render_template('patient/result.html', report=report, doctors=doctors)

@app.route('/patient/history')
def patient_history():
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    user_id = session['user_id']
    # UPDATED QUERY: Fetch prescription as well
    reports = conn.execute('''
        SELECT r.*, c.id as consultation_id, c.doctor_suggestion, c.status, c.prescription
        FROM reports r
        LEFT JOIN consultations c ON r.id = c.report_id
        WHERE r.patient_id = ?
        ORDER BY r.created_at DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return render_template('patient/history.html', reports=reports)

# === CHAT SYSTEM ===
@app.route('/start-chat/<int:report_id>', methods=['POST'])
def start_chat(report_id):
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    doctor_id = request.form['doctor_id']
    message = request.form.get('message', '')
    
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('''SELECT id FROM consultations 
                 WHERE report_id = ? AND patient_id = ?''',
              (report_id, session['user_id']))
    existing = c.fetchone()
    if existing:
        consultation_id = existing[0]
        if message:
            c.execute('''INSERT INTO messages (consultation_id, sender, message)
                         VALUES (?, ?, ?)''', (consultation_id, 'patient', message))
    else:
        c.execute('''INSERT INTO consultations (patient_id, doctor_id, report_id)
                     VALUES (?, ?, ?)''', (session['user_id'], doctor_id, report_id))
        consultation_id = c.lastrowid
        if message:
            c.execute('''INSERT INTO messages (consultation_id, sender, message)
                         VALUES (?, ?, ?)''', (consultation_id, 'patient', message))
    conn.commit()
    conn.close()
    return redirect(f'/chat/{consultation_id}')

@app.route('/chat/<int:consultation_id>')
def chat_room(consultation_id):
    if 'user_id' not in session:
        return redirect('/auth/patient/login')
    
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    consultation = conn.execute('''
        SELECT c.*, p.name as patient_name, d.name as doctor_name, d.specialization as doctor_specialization, r.disease
        FROM consultations c
        JOIN patients p ON c.patient_id = p.id
        JOIN doctors d ON c.doctor_id = d.id
        JOIN reports r ON c.report_id = r.id
        WHERE c.id = ? AND (c.patient_id = ? OR c.doctor_id = ?)
    ''', (consultation_id, session['user_id'], session['user_id'])).fetchone()
    
    if not consultation:
        conn.close()
        return "Access denied", 403
    
    messages = conn.execute('''
        SELECT sender, message, timestamp 
        FROM messages 
        WHERE consultation_id = ? 
        ORDER BY timestamp ASC
    ''', (consultation_id,)).fetchall()
    conn.close()
    
    user_role = session.get('role')
    patient_id = consultation['patient_id']
    doctor_id = consultation['doctor_id']
    
    return render_template('chat.html', 
                           consultation=consultation, 
                           messages=messages, 
                           consultation_id=consultation_id, 
                           user_role=user_role,
                           patient_id=patient_id,
                           doctor_id=doctor_id)

@app.route('/chat/<int:consultation_id>/send', methods=['POST'])
def send_message(consultation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Login required'}), 401
    
    message = request.form['message'].strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('SELECT 1 FROM consultations WHERE id = ? AND (patient_id = ? OR doctor_id = ?)',
              (consultation_id, session['user_id'], session['user_id']))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'Access denied'}), 403
    
    sender = 'doctor' if session.get('role') == 'doctor' else 'patient'
    c.execute('''INSERT INTO messages (consultation_id, sender, message)
                 VALUES (?, ?, ?)''', (consultation_id, sender, message))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/chat/<int:consultation_id>/messages')
def get_messages(consultation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Login required'}), 401
    
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('SELECT 1 FROM consultations WHERE id = ? AND (patient_id = ? OR doctor_id = ?)',
              (consultation_id, session['user_id'], session['user_id']))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'Access denied'}), 403
    
    messages = c.execute('''
        SELECT sender, message, timestamp 
        FROM messages 
        WHERE consultation_id = ? 
        ORDER BY timestamp ASC
    ''', (consultation_id,)).fetchall()
    conn.close()
    return jsonify([{'sender': m[0], 'message': m[1], 'timestamp': m[2]} for m in messages])

@app.route('/consultation/form/<int:report_id>')
def consultation_form(report_id):
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    specializations = conn.execute('SELECT DISTINCT specialization FROM doctors').fetchall()
    conn.close()
    return render_template('patient/consult_form.html', report_id=report_id, specializations=specializations)

@app.route('/get-doctors/<specialization>')
def get_doctors(specialization):
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    doctors = conn.execute('SELECT id, name FROM doctors WHERE specialization = ?', (specialization,)).fetchall()
    conn.close()
    return jsonify([{'id': d['id'], 'name': d['name']} for d in doctors])

@app.route('/consultation/request/<int:report_id>', methods=['POST'])
def request_consultation(report_id):
    if session.get('role') != 'patient':
        return redirect('/auth/patient/login')
    doctor_id = request.form['doctor_id']
    message = request.form.get('message', '')
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('''INSERT INTO consultations (patient_id, doctor_id, report_id, message)
                 VALUES (?, ?, ?, ?)''',
              (session['user_id'], doctor_id, report_id, message))
    conn.commit()
    consultation_id = c.lastrowid       # <--- Get the new ID
    conn.close()
    return redirect(f'/chat/{consultation_id}') # <--- Go STRAIGHT to chat

@app.route('/doctor/dashboard')
def doctor_dashboard():
    if session.get('role') != 'doctor':
        return redirect('/auth/doctor/login')
    return render_template('doctor/dashboard.html')

# === NEW: Doctor Clinical Assessment Page ===
@app.route('/doctor/assess')
def doctor_assess():
    if session.get('role') != 'doctor':
        return redirect('/auth/doctor/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    # Get all patients for the dropdown
    patients = conn.execute('SELECT id, name, email FROM patients').fetchall()
    conn.close()
    return render_template('doctor/assess_patient.html', patients=patients)

@app.route('/doctor/consultations')
def doctor_consultations():
    if session.get('role') != 'doctor':
        return redirect('/auth/doctor/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    consultations = conn.execute('''
        SELECT c.id, c.message, c.doctor_suggestion, c.status, 
               r.disease, r.risk_level, p.name as patient_name
        FROM consultations c
        JOIN reports r ON c.report_id = r.id
        JOIN patients p ON c.patient_id = p.id
        WHERE c.doctor_id = ?
        ORDER BY c.id DESC
    ''', (session['user_id'],)).fetchall()
    consultations = [dict(row) for row in consultations]
    return render_template('doctor/consultations.html', consultations=consultations)

@app.route('/doctor/patient_history')
def doctor_patient_history():
    if session.get('role') != 'doctor':
        return redirect('/auth/doctor/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    consultations = conn.execute('''
        SELECT c.*, r.disease, r.risk_level, p.name as patient_name, r.created_at as report_date
        FROM consultations c
        JOIN reports r ON c.report_id = r.id
        JOIN patients p ON c.patient_id = p.id
        WHERE c.doctor_id = ?
        ORDER BY c.id DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()
    return render_template('doctor/patient_history.html', consultations=consultations)

# === UPDATED: Respond with Prescription ===
@app.route('/consultation/respond/<int:cons_id>', methods=['POST'])
def respond_consultation(cons_id):
    if session.get('role') != 'doctor':
        return redirect('/auth/doctor/login')
    
    suggestion = request.form['suggestion']
    # Capture prescription
    prescription = request.form.get('prescription', '')
    
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('''UPDATE consultations 
                 SET doctor_suggestion = ?, prescription = ?, status = 'responded'
                 WHERE id = ? AND doctor_id = ?''',
              (suggestion, prescription, cons_id, session['user_id']))
    conn.commit()
    conn.close()
    return redirect('/doctor/consultations')

# === NEW: PDF GENERATOR ===
@app.route('/download_prescription/<int:cons_id>')
def download_prescription(cons_id):
    if 'user_id' not in session:
        return redirect('/auth/patient/login')

    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    # Fetch all details needed for PDF
    data = conn.execute('''
        SELECT c.*, p.name as patient_name, d.name as doctor_name, r.disease, r.risk_level
        FROM consultations c
        JOIN patients p ON c.patient_id = p.id
        JOIN doctors d ON c.doctor_id = d.id
        JOIN reports r ON c.report_id = r.id
        WHERE c.id = ? AND (c.patient_id = ? OR c.doctor_id = ?)
    ''', (cons_id, session['user_id'], session['user_id'])).fetchone()
    conn.close()

    if not data or not data['prescription']:
        return "Prescription not found", 404

    # Generate PDF using FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="HealthBridge Prescription", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Doctor: {data['doctor_name']}", ln=True)
    pdf.cell(200, 10, txt=f"Patient: {data['patient_name']}", ln=True)
    pdf.cell(200, 10, txt=f"Condition: {data['disease']} (Risk: {data['risk_level']})", ln=True)
    pdf.ln(10)
    
    # Body
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Rx / Prescription:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=data['prescription'])
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="This is a digitally generated prescription.", ln=True)

    # Output to buffer
    output = io.BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin-1') 
    output.write(pdf_content)
    output.seek(0)
    
    return send_file(output, as_attachment=True, download_name=f"prescription_{cons_id}.pdf", mimetype='application/pdf')

# === ADMIN MODULE ===
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['role'] = 'admin'
            return redirect('/admin/dashboard')
        return render_template('admin/login.html', error="Invalid credentials")
    return render_template('admin/login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    return render_template('admin/dashboard.html')

@app.route('/admin/patients')
def admin_patients():
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    patients = conn.execute('SELECT * FROM patients').fetchall()
    conn.close()
    return render_template('admin/patients.html', patients=patients)

@app.route('/admin/doctors')
def admin_doctors():
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    doctors = conn.execute('SELECT * FROM doctors WHERE email != "doctor@example.com"').fetchall()
    conn.close()
    return render_template('admin/doctors.html', doctors=doctors)

@app.route('/admin/reports')
def admin_reports():
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    reports = conn.execute('''
        SELECT r.id, r.disease, r.risk_level, r.mode, r.created_at,
               p.name as patient_name, d.name as doctor_name
        FROM reports r
        LEFT JOIN patients p ON r.patient_id = p.id
        LEFT JOIN consultations c ON r.id = c.report_id
        LEFT JOIN doctors d ON c.doctor_id = d.id
        ORDER BY r.created_at DESC
    ''').fetchall()
    conn.close()
    return render_template('admin/reports.html', reports=reports)

@app.route('/admin/patient/add', methods=['GET', 'POST'])
def admin_add_patient():
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('healthbridge.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO patients (name, email, password) VALUES (?, ?, ?)', (name, email, password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('admin/add_patient.html', error="Email exists")
        conn.close()
        return redirect('/admin/patients')
    return render_template('admin/add_patient.html')

@app.route('/admin/doctor/add', methods=['GET', 'POST'])
def admin_add_doctor():
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        specialization = request.form['specialization']
        conn = sqlite3.connect('healthbridge.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO doctors (name, email, password, specialization) VALUES (?, ?, ?, ?)',
                      (name, email, password, specialization))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('admin/add_doctor.html', error="Email exists")
        conn.close()
        return redirect('/admin/doctors')
    return render_template('admin/add_doctor.html')

@app.route('/admin/patient/edit/<int:patient_id>', methods=['GET', 'POST'])
def admin_edit_patient(patient_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        query = 'UPDATE patients SET name = ?, email = ?'
        params = [name, email]
        if password:
            query += ', password = ?'
            params.append(generate_password_hash(password))
        query += ' WHERE id = ?'
        params.append(patient_id)
        c = conn.cursor()
        c.execute(query, params)
        conn.commit()
        conn.close()
        return redirect('/admin/patients')
    conn.close()
    return render_template('admin/edit_patient.html', patient=patient)

@app.route('/admin/doctor/edit/<int:doctor_id>', methods=['GET', 'POST'])
def admin_edit_doctor(doctor_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    conn.row_factory = sqlite3.Row
    doctor = conn.execute('SELECT * FROM doctors WHERE id = ? AND email != "doctor@example.com"', (doctor_id,)).fetchone()
    if not doctor:
        return "Access denied", 403
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        specialization = request.form['specialization']
        query = 'UPDATE doctors SET name = ?, email = ?, specialization = ?'
        params = [name, email, specialization]
        if password:
            query += ', password = ?'
            params.append(generate_password_hash(password))
        query += ' WHERE id = ?'
        params.append(doctor_id)
        c = conn.cursor()
        c.execute(query, params)
        conn.commit()
        conn.close()
        return redirect('/admin/doctors')
    conn.close()
    return render_template('admin/edit_doctor.html', doctor=doctor)

@app.route('/admin/report/delete/<int:report_id>', methods=['POST'])
def admin_delete_report(report_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
    return redirect('/admin/reports')

@app.route('/admin/patient/delete/<int:patient_id>', methods=['POST'])
def admin_delete_patient(patient_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
    conn.commit()
    conn.close()
    return redirect('/admin/patients')

@app.route('/admin/doctor/delete/<int:doctor_id>', methods=['POST'])
def admin_delete_doctor(doctor_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')
    conn = sqlite3.connect('healthbridge.db')
    c = conn.cursor()
    c.execute('DELETE FROM doctors WHERE id = ? AND email != "doctor@example.com"', (doctor_id,))
    conn.commit()
    conn.close()
    return redirect('/admin/doctors')

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect('/admin/login')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3012, debug=True)
