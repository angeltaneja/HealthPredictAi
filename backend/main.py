import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our RAG query function
# We import it here so that it loads when needed
from rag_pipeline import query_patient_assistant

load_dotenv()

app = FastAPI(title="HealthPredictAI API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the Vite origin e.g. http://localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model and data
model_pipeline = None
feature_metadata = None
patients_df = None

def load_resources():
    global model_pipeline, feature_metadata, patients_df
    
    # Load patient data
    csv_path = "data/patients.csv"
    if os.path.exists(csv_path):
        patients_df = pd.read_csv(csv_path)
    else:
        print(f"Warning: {csv_path} not found. Run generate_data.py first.")
        
    # Load ML model
    model_path = "models/readmission_pipeline.pkl"
    if os.path.exists(model_path):
        with open(model_path, "wb" if not os.path.exists(model_path) else "rb") as f:
            model_pipeline = pickle.load(f)
    else:
        print(f"Warning: {model_path} not found. Run train_model.py first.")
        
    # Load feature metadata
    meta_path = "models/feature_importances.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            feature_metadata = pickle.load(f)

# Initialize resources on startup
@app.on_event("startup")
def startup_event():
    load_resources()

class ChatRequest(BaseModel):
    query: str

@app.get("/api/patients")
def get_patients():
    global patients_df
    if patients_df is None:
        load_resources()
    if patients_df is None:
        raise HTTPException(status_code=500, detail="Patient dataset not initialized.")
        
    # Return basic fields for the list view
    patient_list = patients_df[[
        "PatientID", "Age", "Gender", "BMI", "PrimaryDiagnosis", 
        "LengthOfStay", "PriorAdmissions", "Comorbidities", 
        "DischargeDisposition", "Readmitted"
    ]].to_dict(orient="records")
    
    return patient_list

@app.get("/api/patients/{patient_id}/risk")
def get_patient_risk(patient_id: str):
    global patients_df, model_pipeline, feature_metadata
    if patients_df is None or model_pipeline is None:
        load_resources()
        
    if patients_df is None:
        raise HTTPException(status_code=500, detail="Patient dataset not initialized.")
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="ML model pipeline not initialized.")
        
    # Get patient record
    patient_record = patients_df[patients_df["PatientID"] == patient_id]
    if patient_record.empty:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    # Extract features for prediction (dropping PatientID, ReadmitProbability, Readmitted)
    X_patient = patient_record.drop(columns=["PatientID", "ReadmitProbability", "Readmitted"])
    
    # Run prediction
    risk_proba = model_pipeline.predict_proba(X_patient)[0][1]
    risk_percent = round(float(risk_proba) * 100, 1)
    
    # Calculate personalized risk factors (explainability)
    # Compare patient's feature values to typical/healthy thresholds
    # to find which factors contributed the most to their risk
    row = patient_record.iloc[0]
    factors = []
    
    if row["PriorAdmissions"] > 0:
        factors.append({
            "factor": "Prior Admissions",
            "value": f"{row['PriorAdmissions']} in last 12m",
            "impact": "High",
            "score": round(row["PriorAdmissions"] * 0.8, 2)
        })
        
    if row["Comorbidities"] > 0:
        factors.append({
            "factor": "Comorbidities",
            "value": f"{row['Comorbidities']} active diagnoses",
            "impact": "Medium" if row["Comorbidities"] < 3 else "High",
            "score": round(row["Comorbidities"] * 0.5, 2)
        })
        
    if row["LabCreatinine"] > 1.2:
        factors.append({
            "factor": "Abnormal Kidney Function (Creatinine)",
            "value": f"{row['LabCreatinine']} mg/dL",
            "impact": "High",
            "score": round((row["LabCreatinine"] - 1.2) * 1.5, 2)
        })
        
    if row["LabHbA1c"] > 7.0:
        factors.append({
            "factor": "Uncontrolled Glycemia (HbA1c)",
            "value": f"{row['LabHbA1c']}%",
            "impact": "Medium",
            "score": round((row["LabHbA1c"] - 7.0) * 0.4, 2)
        })
        
    if row["LengthOfStay"] > 5:
        factors.append({
            "factor": "Extended Hospital Stay",
            "value": f"{row['LengthOfStay']} days",
            "impact": "Medium",
            "score": round((row["LengthOfStay"] - 5) * 0.15, 2)
        })
    elif row["LengthOfStay"] <= 2:
        factors.append({
            "factor": "Short Hospital Stay (Potential Premature Discharge)",
            "value": f"{row['LengthOfStay']} days",
            "impact": "Medium",
            "score": 0.3
        })
        
    if row["DischargeDisposition"] in ["Skilled Nursing Facility", "Rehab"]:
        factors.append({
            "factor": "Discharged to Facility",
            "value": row["DischargeDisposition"],
            "impact": "High",
            "score": 1.0
        })
    elif row["DischargeDisposition"] == "Home Health Care":
        factors.append({
            "factor": "Discharged with Home Care",
            "value": "Home Health Care",
            "impact": "Low",
            "score": 0.3
        })
        
    if row["Age"] > 65:
        factors.append({
            "factor": "Advanced Age",
            "value": f"{row['Age']} years",
            "impact": "Low",
            "score": round((row["Age"] - 50) * 0.02, 2)
        })
        
    # Sort factors by impact score
    factors = sorted(factors, key=lambda x: x["score"], reverse=True)
    
    return {
        "PatientID": patient_id,
        "ReadmissionRisk": risk_percent,
        "PrimaryDiagnosis": row["PrimaryDiagnosis"],
        "RiskFactors": factors[:4]  # Return top 4 factors
    }

@app.post("/api/patients/{patient_id}/chat")
def chat_patient_record(patient_id: str, request: ChatRequest):
    global patients_df
    if patients_df is None:
        load_resources()
    if patients_df is None:
        raise HTTPException(status_code=500, detail="Patient dataset not initialized.")
        
    patient_record = patients_df[patients_df["PatientID"] == patient_id]
    if patient_record.empty:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    try:
        response = query_patient_assistant(patient_id, request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
