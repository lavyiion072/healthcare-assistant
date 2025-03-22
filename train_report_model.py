import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('report_data.csv')  # CSV in .txt format

# Vectorize report_text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['report_text'])
y = df['Disease']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and vectorizer
with open('report_diagnosis_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Trained and saved model + vectorizer.")
