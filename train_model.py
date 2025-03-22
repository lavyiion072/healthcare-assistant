import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv('disease_data.csv')  # Replace with your actual file name

# Extract features and labels
X = df.drop('disease', axis=1)
y = df['disease']

# Create symptom index
symptom_index = {symptom.lower(): idx for idx, symptom in enumerate(X.columns)}

# Create disease list
disease_list = list(np.unique(y))

# Encode disease labels as indices
disease_to_index = {disease: idx for idx, disease in enumerate(disease_list)}
index_to_disease = {idx: disease for disease, idx in disease_to_index.items()}
y_encoded = y.map(disease_to_index)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X.values, y_encoded)

# Save the model
with open('disease_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the disease list (in order of index)
with open('disease_list.pkl', 'wb') as f:
    pickle.dump(index_to_disease, f)

# Save the symptom index
with open('symptom_index.pkl', 'wb') as f:
    pickle.dump(symptom_index, f)

print("✅ Model and files created successfully.")
