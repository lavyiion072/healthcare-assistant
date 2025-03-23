from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import UserMixin

db = SQLAlchemy()

# -------------------- Doctor Model -------------------- #
class Doctor(db.Model, UserMixin):  # Enables login functionality
    id = db.Column(db.Integer, primary_key=True)
    
    # Basic Details
    name = db.Column(db.String(100), nullable=False)
    designation = db.Column(db.String(100), nullable=True)
    specialization = db.Column(db.String(100), nullable=True)
    degree = db.Column(db.String(100), nullable=True)

    # Availability Timings
    start_time = db.Column(db.String(5), nullable=True)  # e.g., "09:00"
    end_time = db.Column(db.String(5), nullable=True)    # e.g., "17:30"
    availability = db.Column(db.String(200), nullable=True)  # e.g., "Monday,Wednesday,Friday"

    # Location Info
    service_area = db.Column(db.String(200), nullable=True)      # e.g., "Ahmedabad"
    clinic_location = db.Column(db.String(300), nullable=True)   # e.g., "Gota, Ahmedabad"
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

    # Appointments
    appointment_available = db.Column(db.Boolean, default=False)

    # Fees
    standard_fee = db.Column(db.Float, nullable=True)
    emergency_fee = db.Column(db.Float, nullable=True)

    # Contact + Login
    mobile = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # Should be hashed!
    role = db.Column(db.String(20), default='doctor')     # Default role for doctors

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Doctor {self.name}>"

# -------------------- Laboratory Model -------------------- #
class Laboratory(db.Model, UserMixin):  # Enables login functionality
    id = db.Column(db.Integer, primary_key=True)

    # Lab Info
    lab_name = db.Column(db.String(100), nullable=False)                   # Lab Name
    lab_head = db.Column(db.String(100), nullable=False)                   # Lab Head Name
    specialization = db.Column(db.String(100), nullable=False)            # e.g., Pathology, Radiology

    # Availability Timings
    start_time = db.Column(db.String(5), nullable=False)                   # e.g., "09:00"
    end_time = db.Column(db.String(5), nullable=False)                     # e.g., "18:30"
    weekly_availability = db.Column(db.String(200), nullable=False)       # e.g., "Monday,Tuesday,Friday"

    # Location Info
    service_location = db.Column(db.String(200), nullable=False)          # e.g., "Ahmedabad"
    lab_address = db.Column(db.String(300), nullable=False)               # Detailed Address
    latitude = db.Column(db.Float, nullable=False)                        # From map input
    longitude = db.Column(db.Float, nullable=False)                       # From map input

    # Services
    home_sample_available = db.Column(db.Boolean, default=False)          # Home sample collection

    # Fees
    standard_fee = db.Column(db.Integer, nullable=False)                  # ₹ Standard Fee
    emergency_fee = db.Column(db.Integer, nullable=True)                  # ₹ Emergency Fee (optional)

    # Contact + Login
    mobile = db.Column(db.String(15), unique=True, nullable=False)        # Contact Number
    password = db.Column(db.String(255), nullable=False)                  # Hashed password
    role = db.Column(db.String(20), default='laboratory')                 # Default role for labs

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Laboratory {self.lab_name} - Head: {self.lab_head}>"

# -------------------- User (Patient) Model -------------------- #
class User(db.Model, UserMixin):  # Enables login for patients
    id = db.Column(db.Integer, primary_key=True)

    # Personal Info
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    mobile = db.Column(db.String(15), unique=True, nullable=False)
    location = db.Column(db.String(200), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

    # Medical Info
    blood_group = db.Column(db.String(5), nullable=True)
    last_disease = db.Column(db.String(200), nullable=True)
    other_health_data = db.Column(db.Text, nullable=True)

    # Recommendations
    recommended_doctor = db.Column(db.String(100), nullable=True)
    recommended_laboratory = db.Column(db.String(100), nullable=True)

    # Contact + Login
    password = db.Column(db.String(200), nullable=False)  # Hashed password
    role = db.Column(db.String(20), default='user')       # Default role for patients

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.name}>"
