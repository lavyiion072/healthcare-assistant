from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import fitz  # PyMuPDF
import pickle
from difflib import get_close_matches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import secrets

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from models import db, Doctor, Laboratory, User

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# --- Configurations ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Load and Train Models ---
df = pd.read_csv('medicine_data.csv')

symptom_enc = LabelEncoder()
age_enc = LabelEncoder()
severity_enc = LabelEncoder()

df['symptom_enc'] = symptom_enc.fit_transform(df['Symptom'].str.lower())
df['age_enc'] = age_enc.fit_transform(df['AgeGroup'].str.lower())
df['severity_enc'] = severity_enc.fit_transform(df['Severity'].str.lower())
df['duration_days'] = df['Duration']

X = df[['symptom_enc', 'age_enc', 'severity_enc', 'duration_days']]
y = df.index

medicine_model = RandomForestClassifier()
medicine_model.fit(X, y)

disease_model = pickle.load(open("disease_prediction_model.pkl", "rb"))
disease_list = pickle.load(open("disease_list.pkl", "rb"))
symptom_index = pickle.load(open("symptom_index.pkl", "rb"))

report_model = pickle.load(open("report_diagnosis_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --- Helper Functions ---
def suggest_medicines(symptom, age_group, severity, duration):
    known_symptoms = list(symptom_enc.classes_)
    matched = get_close_matches(symptom.lower(), known_symptoms, n=1, cutoff=0.6)
    matched_symptom = matched[0] if matched else 'fever'
    try:
        sym_val = symptom_enc.transform([matched_symptom])[0]
        age_val = age_enc.transform([age_group.lower()])[0]
        sev_val = severity_enc.transform([severity.lower()])[0]

        input_df = pd.DataFrame([{
            'symptom_enc': sym_val,
            'age_enc': age_val,
            'severity_enc': sev_val,
            'duration_days': duration
        }])

        pred_idx = medicine_model.predict(input_df)[0]
        row = df.loc[pred_idx]

        result = {
            'medicine_1': row['medicine_1'],
            'company_1': row['company_1'],
            'description_1': row['description_1'],
            'dosage_1': row['dosage_1'],
            'course_days_1': row['course_days_1'],
            'price_1': row['price_1'],
            'quantity_1': row['quantity_1'],
            'medicine_2': row['medicine_2'],
            'company_2': row['company_2'],
            'description_2': row['description_2'],
            'dosage_2': row['dosage_2'],
            'course_days_2': row['course_days_2'],
            'price_2': row['price_2'],
            'quantity_2': row['quantity_2'],
            'recommended_tests': row['tests_suggested'] if pd.notna(row['tests_suggested']) else 'No tests recommended'
        }
        return result
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

def predict_disease(symptoms):
    input_data = [0] * len(symptom_index)
    all_known_symptoms = list(symptom_index.keys())

    for symptom in symptoms:
        symptom = symptom.strip().lower()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
        else:
            close_match = get_close_matches(symptom, all_known_symptoms, n=1, cutoff=0.6)
            if close_match:
                input_data[symptom_index[close_match[0]]] = 1
            else:
                print(f"Unknown symptom ignored: {symptom}")

    prediction = disease_model.predict([input_data])[0]
    return disease_list[prediction]

def extract_text_from_pdf(file_stream):
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def predict_from_report(text):
    transformed_text = vectorizer.transform([text])
    prediction = report_model.predict(transformed_text)[0]
    return prediction

# --- Login Loader ---
@login_manager.user_loader
def load_user(user_id):
    role = session.get('role')
    if role == 'doctor':
        return Doctor.query.get(int(user_id))
    elif role == 'laboratory':
        return Laboratory.query.get(int(user_id))
    else:
        return User.query.get(int(user_id))

# --- Index Route ---
@app.route('/prediction-model', methods=['GET', 'POST'])
def index():
    medicine_result = None
    disease_result = None
    report_result = None
    error = None
    active_tab = 0

    if request.method == 'POST':
        form_type = request.form.get('form_type')
        if form_type == 'medicine':
            active_tab = 0
            try:
                symptom = request.form.get('symptom', '').strip()
                age_group = request.form.get('age_group', '').strip()
                severity = request.form.get('severity', '').strip()
                duration_input = request.form.get('duration', '').strip()

                if not (symptom and age_group and severity and duration_input.isdigit()):
                    raise ValueError("❌ Invalid input. Please fill all fields correctly.")

                duration = int(duration_input)
                suggestion = suggest_medicines(symptom, age_group, severity, duration)
                medicine_result = suggestion if suggestion else None
                if not suggestion:
                    error = "❌ Unable to find a match for the given input."
            except Exception as e:
                print(f"Medicine Form Error: {e}")
                error = f"❌ Something went wrong. {str(e)}"

        elif form_type == 'disease':
            active_tab = 1
            try:
                symptoms_input = request.form.get('symptoms', '').strip()
                if not symptoms_input:
                    raise ValueError("❌ Symptoms cannot be empty.")
                symptoms_list = [s.strip() for s in symptoms_input.split(',') if s.strip()]
                disease_result = predict_disease(symptoms_list)
            except Exception as e:
                print(f"Disease Prediction Error: {e}")
                error = f"❌ Could not predict the disease. {str(e)}"

        elif form_type == 'report':
            active_tab = 2
            try:
                file = request.files.get('report')
                if not file:
                    raise ValueError("❌ No file uploaded.")
                text = extract_text_from_pdf(file)
                if not text:
                    raise ValueError("❌ No text extracted from PDF.")
                report_result = predict_from_report(text)
            except Exception as e:
                print(f"Report Prediction Error: {e}")
                error = f"❌ Could not process the uploaded report. {str(e)}"

    return render_template(
        'index.html',
        medicine_result=medicine_result,
        disease_result=disease_result,
        report_diagnosis=report_result,
        error=error,
        active_tab=active_tab
    )

# --- Login ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        mobile = request.form['mobile']
        password = request.form['password']

        user = User.query.filter_by(mobile=mobile).first()
        doctor = Doctor.query.filter_by(mobile=mobile).first()
        lab = Laboratory.query.filter_by(mobile=mobile).first()

        if doctor and bcrypt.check_password_hash(doctor.password, password):
            login_user(doctor)
            session['role'] = 'doctor'
            flash('Doctor login successful!', 'success')
            return redirect(url_for('doctor_dashboard'))

        elif lab and bcrypt.check_password_hash(lab.password, password):
            login_user(lab)
            session['role'] = 'laboratory'
            flash('Laboratory login successful!', 'success')
            return redirect(url_for('lab_dashboard'))

        elif user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            session['role'] = 'user'
            flash('User login successful!', 'success')
            return redirect(url_for('user_dashboard'))

        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('role', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))
 
# --- Registration with Validations ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password', '').strip()
        designation = request.form.get('designation', '').strip()
        specialization = request.form.get('specialization', '').strip()
        degree = request.form.get('degree', '').strip()
        start_time = request.form.get('start_time', '').strip()
        end_time = request.form.get('end_time', '').strip()
        availability = request.form.get('availability', '').strip()
        service_area = request.form.get('service_location', '').strip()
        clinic_location = request.form.get('clinic_location', '').strip()
        latitude = request.form.get('latitude', '').strip()
        longitude = request.form.get('longitude', '').strip()
        appointment_available = request.form.get('appointment_available', '').strip()
        standard_fee = request.form.get('standard_fee', '').strip()
        emergency_fee = request.form.get('emergency_fee', '').strip()
        role = request.form.get('form_type', '').strip()
        
        errors = []

        # --- Basic Validations ---
        if not name:
            errors.append("Name is required.")
        if not mobile or not mobile.isdigit() or len(mobile) != 10:
            errors.append("Valid 10-digit mobile number is required.")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if role not in ['doctor', 'laboratory', 'user']:
            errors.append("Invalid role selected.")

        # --- Doctor-Specific Validations ---
        if role == 'doctor':
            required_fields = [designation, specialization, degree, start_time, end_time,
                               availability, service_area, clinic_location, latitude, longitude,
                               appointment_available, standard_fee, emergency_fee]
            if not all(required_fields):
                errors.append("All fields are required for doctor registration.")
            try:
                float(latitude)
                float(longitude)
                float(standard_fee)
                float(emergency_fee)
            except ValueError:
                errors.append("Latitude, longitude, and fees must be valid numbers.")

        # --- If Errors, Show Messages ---
        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')  # Keep the filled form if needed

        # --- If Valid, Proceed with Registration ---
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            if role == 'doctor':
                new_user = Doctor(
                    name=name, mobile=mobile, password=hashed_pw, 
                    designation=designation, specialization=specialization, 
                    degree=degree, start_time=start_time, end_time=end_time, 
                    availability=availability, service_area=service_area, 
                    clinic_location=clinic_location, latitude=latitude, 
                    longitude=longitude, appointment_available=appointment_available, 
                    standard_fee=standard_fee, emergency_fee=emergency_fee
                )
            elif role == 'laboratory':
                new_user = Laboratory(name=name, mobile=mobile, password=hashed_pw)
            else:
                new_user = User(name=name, mobile=mobile, password=hashed_pw)

            db.session.add(new_user)
            db.session.commit()
            flash(f'{role.capitalize()} registered successfully!', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            print(f"Registration Error: {e}")
            flash('Error during registration. Try again.', 'danger')

    return render_template('register.html')

# --- Dashboards ---
@app.route('/doctor_dashboard')
@login_required
def doctor_dashboard():
    if session.get('role') != 'doctor':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('login'))
    doctor = current_user
    session['doctor_id'] = doctor.id
    return render_template('doctor_dashboard.html', doctor=doctor)

@app.route('/doctor_dashboard', methods=['GET', 'POST'])
@login_required
def update_profile():
    doctor_id = session['doctor_id']
    doctor = Doctor.query.get(doctor_id)
    if not doctor:
        flash('Doctor not found!')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        doctor.name = request.form['name']
        doctor.designation = request.form['designation']
        doctor.specialization = request.form['specialization']
        doctor.degree = request.form['degree']
        doctor.start_time = request.form['start_time']
        doctor.end_time = request.form['end_time']
        appointment_value = request.form.get('appointment_available')
        doctor.appointment_available = True if appointment_value == 'on' else False
        availability_list = request.form.getlist('availability[]')
        availability_str = ",".join(availability_list)
        doctor.availability = availability_str
        doctor.service_area = request.form['service_location']
        doctor.clinic_location = request.form['clinic_location']
        doctor.latitude = request.form['latitude']
        doctor.longitude = request.form['longitude']
        doctor.standard_fee = request.form['standard_fee']
        doctor.emergency_fee = request.form['emergency_fee']

        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('doctor_dashboard'))
    return render_template('doctor_dashboard.html', doctor=doctor)


@app.route('/lab_dashboard')
@login_required
def lab_dashboard():
    if session.get('role') != 'laboratory':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('login'))
    return render_template('lab_dashboard.html')

@app.route('/user_dashboard')
@login_required
def user_dashboard():
    if session.get('role') != 'user':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('login'))
    return render_template('user_dashboard.html')

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
