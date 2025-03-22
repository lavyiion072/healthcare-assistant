from flask import Flask, render_template, request
import pandas as pd
import fitz  # PyMuPDF
import pickle
from difflib import get_close_matches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- Medicine Suggestion Setup (same as before) ---
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
model = RandomForestClassifier()
model.fit(X, y)

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

        pred_idx = model.predict(input_df)[0]
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

# --- Disease Prediction by Symptoms ---
disease_model = pickle.load(open("disease_prediction_model.pkl", "rb"))
disease_list = pickle.load(open("disease_list.pkl", "rb"))
symptom_index = pickle.load(open("symptom_index.pkl", "rb"))

def predict_disease(symptoms):
    input_data = [0] * len(symptom_index)
    all_known_symptoms = list(symptom_index.keys())

    for symptom in symptoms:
        symptom = symptom.strip().lower()
        if symptom in symptom_index:
            index = symptom_index[symptom]
            input_data[index] = 1
        else:
            close_match = get_close_matches(symptom, all_known_symptoms, n=1, cutoff=0.6)
            if close_match:
                matched_symptom = close_match[0]
                index = symptom_index[matched_symptom]
                input_data[index] = 1
            else:
                print(f"Unknown symptom ignored: {symptom}")

    prediction = disease_model.predict([input_data])[0]
    return disease_list[prediction]

# --- Disease Prediction by Uploaded PDF Report ---
report_model = pickle.load(open("report_diagnosis_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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

# --- Unified Route for Forms ---
@app.route('/', methods=['GET', 'POST'])
def index():
    medicine_result = None
    disease_result = None
    report_result = None
    error = None
    active_tab = 0

    if request.method == 'POST':
        form_type = request.form.get('form_type')
        print(form_type)

        if form_type == 'medicine':
            active_tab = 0
            try:
                symptom = request.form.get('symptom', '').strip()
                age_group = request.form.get('age_group', '').strip()
                severity = request.form.get('severity', '').strip()
                duration_input = request.form.get('duration', '').strip()

                # Enhanced Input Validation
                if not (symptom and age_group and severity and duration_input.isdigit()):
                    raise ValueError("❌ Invalid input. Please make sure all fields are filled correctly.")
                
                duration = int(duration_input)
                suggestion = suggest_medicines(symptom, age_group, severity, duration)
                if suggestion:
                    medicine_result = suggestion
                else:
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
                print (disease_result)
            except Exception as e:
                print(f"Disease Prediction Error: {e}")
                error = f"❌ Could not predict the disease. {str(e)}"

        elif form_type == 'report':
            active_tab = 2
            try:
                print(request.files)
                file = request.files.get('report')
                if not file:
                    raise ValueError("❌ No file uploaded.")
                text = extract_text_from_pdf(file)
                if not text:
                    raise ValueError("❌ No text extracted from PDF.")
                report_result = predict_from_report(text)
                print(report_result)
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

if __name__ == '__main__':
    app.run(debug=True)
