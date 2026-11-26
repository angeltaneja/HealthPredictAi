import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_patients=1000):
    print(f"Generating {num_patients} patient records...")
    
    # 1. Generate core features
    patient_ids = [f"PAT_{i:04d}" for i in range(1, num_patients + 1)]
    ages = np.random.randint(18, 90, size=num_patients)
    genders = np.random.choice(["Male", "Female"], size=num_patients, p=[0.48, 0.52])
    bmis = np.round(np.random.normal(27.5, 5.0, size=num_patients), 1)
    
    diagnoses = ["Heart Failure", "Diabetes", "COPD", "Pneumonia", "Hypertension", "Stroke"]
    primary_diagnoses = np.random.choice(diagnoses, size=num_patients, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    
    length_of_stay = np.random.geometric(p=0.25, size=num_patients) + 1  # Mean stay ~ 5 days
    length_of_stay = np.clip(length_of_stay, 1, 14)  # Bound stay between 1 and 14 days
    
    prior_admissions = np.random.poisson(lam=0.8, size=num_patients)
    comorbidities = np.random.poisson(lam=1.5, size=num_patients)
    
    dispositions = ["Home", "Home Health Care", "Skilled Nursing Facility", "Rehab"]
    discharge_dispositions = np.random.choice(dispositions, size=num_patients, p=[0.55, 0.25, 0.15, 0.05])
    
    # Lab values
    lab_hbA1c = np.round(np.random.normal(6.5, 1.5, size=num_patients), 1)
    lab_creatinine = np.round(np.random.normal(1.1, 0.4, size=num_patients), 2)
    lab_sodium = np.round(np.random.normal(138, 4.0, size=num_patients), 1)
    
    admit_sources = ["Emergency Room", "Referral", "Urgent Care"]
    admit_sources_chosen = np.random.choice(admit_sources, size=num_patients, p=[0.60, 0.25, 0.15])
    
    # 2. Determine readmission risk probability based on clinical factors
    # We want to create realistic predictive signals for the ML model:
    # - More prior admissions increases risk
    # - More comorbidities increases risk
    # - Length of stay: very short or very long increases risk
    # - Discharge to Skilled Nursing or Rehab increases risk
    # - Abnormal labs (high HbA1c, high Creatinine, low/high Sodium) increases risk
    # - Older age increases risk
    
    logit = (
        -4.2  # baseline
        + 1.8 * prior_admissions
        + 1.2 * comorbidities
        + 0.05 * (ages - 50)
        + 0.3 * (length_of_stay - 5)
        + 1.5 * (discharge_dispositions == "Skilled Nursing Facility").astype(int)
        + 1.0 * (discharge_dispositions == "Rehab").astype(int)
        + 0.8 * np.maximum(0, lab_hbA1c - 7.0)
        + 2.0 * np.maximum(0, lab_creatinine - 1.2)
        + 0.2 * np.abs(lab_sodium - 138.0)
    )
    
    probabilities = 1 / (1 + np.exp(-logit))
    # Thresholding with 2% random noise to make it realistic but achieve ~93% accuracy
    readmitted = (probabilities > 0.5).astype(int)
    noise_mask = np.random.rand(num_patients) < 0.02
    readmitted[noise_mask] = 1 - readmitted[noise_mask]
    
    # Create DataFrame
    df = pd.DataFrame({
        "PatientID": patient_ids,
        "Age": ages,
        "Gender": genders,
        "BMI": bmis,
        "PrimaryDiagnosis": primary_diagnoses,
        "LengthOfStay": length_of_stay,
        "PriorAdmissions": prior_admissions,
        "Comorbidities": comorbidities,
        "DischargeDisposition": discharge_dispositions,
        "LabHbA1c": lab_hbA1c,
        "LabCreatinine": lab_creatinine,
        "LabSodium": lab_sodium,
        "AdmitSource": admit_sources_chosen,
        "ReadmitProbability": np.round(probabilities, 3),
        "Readmitted": readmitted
    })
    
    # Save CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/patients.csv", index=False)
    print("Saved data/patients.csv")
    return df

def generate_discharge_summaries(df):
    print("Generating clinical discharge summaries...")
    summaries_dir = "data/discharge_summaries"
    os.makedirs(summaries_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        pid = row["PatientID"]
        age = row["Age"]
        gender = row["Gender"]
        dx = row["PrimaryDiagnosis"]
        los = row["LengthOfStay"]
        disp = row["DischargeDisposition"]
        priors = row["PriorAdmissions"]
        comorb = row["Comorbidities"]
        hba1c = row["LabHbA1c"]
        creat = row["LabCreatinine"]
        sod = row["LabSodium"]
        
        pronoun_sub = "He" if gender == "Male" else "She"
        pronoun_poss = "His" if gender == "Male" else "Her"
        pronoun_obj = "him" if gender == "Male" else "her"
        
        # Medical description based on primary diagnosis
        dx_details = ""
        course_details = ""
        plan_details = ""
        
        if dx == "Heart Failure":
            dx_details = f"congestive heart failure (CHF) and exacerbation of chronic shortness of breath. The patient presented with progressive dyspnea, baseline lower extremity edema, and orthopnea."
            course_details = f"The patient was initiated on aggressive intravenous diuresis (Furosemide 80mg IV). Input and output volumes were strictly monitored. BNP level was elevated. By day {los // 2 + 1}, the patient showed substantial weight reduction, improved oxygenation on room air, and resolution of peripheral edema."
            plan_details = f"Continue oral Furosemide (Lasix) 40mg daily. Daily weight monitoring is crucial. Follow up with Cardiology clinic within 7 days. Restrict dietary sodium to less than 2,000 mg daily."
        elif dx == "Diabetes":
            dx_details = f"uncontrolled Type 2 Diabetes Mellitus with symptomatic hyperglycemia. The patient reported polyuria, polydipsia, and mild fatigue over the last two weeks."
            course_details = f"Fluid resuscitation was completed. The patient was started on a sliding scale insulin regimen, which was subsequently transitioned to a long-acting insulin (Lantus) combined with Metformin. HbA1c was measured at {hba1c}%. Nutritional services provided counseling on low glycemic diet management."
            plan_details = f"Administer Lantus 15 units subcutaneously at bedtime. Metformin 1000mg twice daily with meals. Monitor blood glucose before meals and at bedtime. Follow up with Endocrinology and primary care."
        elif dx == "COPD":
            dx_details = f"acute exacerbation of Chronic Obstructive Pulmonary Disease (COPD). The patient presented with severe wheezing, productive cough with purulent sputum, and hypoxia."
            course_details = f"Oxygen therapy was titrated to maintain saturation between 88-92%. The patient was treated with intravenous methylprednisolone, scheduled bronchodilator nebulizers (Albuterol/Ipratropium), and oral Azithromycin for suspected bacterial infection. Sputum cultures were negative."
            plan_details = f"Taper prednisone over 5 days. Continue Symbicort inhaler 2 puffs twice daily and Albuterol rescue inhaler as needed. Avoid known airway irritants. Scheduled follow-up with Pulmonology."
        elif dx == "Pneumonia":
            dx_details = f"community-acquired lobar pneumonia. The patient presented with high fever, chills, pleuritic chest pain, and a productive cough. Chest X-ray confirmed right lower lobe consolidation."
            course_details = f"Empirical antibiotic therapy was initiated with Ceftriaxone and Azithromycin. Sputum and blood cultures were obtained. The patient remained febrile for the first 36 hours but gradually defervesced. Oxygen requirements decreased, and leucocytosis resolved by day {los}."
            plan_details = f"Complete the oral course of Levofloxacin 500mg daily for another 5 days. Rest and deep breathing exercises. Repeat chest X-ray in 6 weeks to document clearance. Close monitoring for recurrence of fever or chest pain."
        elif dx == "Hypertension":
            dx_details = f"hypertensive emergency. The patient presented with severe headache, blurred vision, and a blood pressure reading of 190/115 mmHg."
            course_details = f"Intravenous Labetalol was administered in the emergency department for controlled blood pressure reduction. Subsequently, the patient was transitioned to oral Lisinopril and Amlodipine. Renal function was assessed (Creatinine was {creat} mg/dL) to rule out acute kidney injury. Blood pressure stabilized below 135/85 mmHg."
            plan_details = f"Take Lisinopril 20mg daily and Amlodipine 5mg daily. Check blood pressure twice daily and log results. Avoid NSAIDs. Urgent primary care follow-up scheduled for medication optimization."
        else:  # Stroke
            dx_details = f"acute ischemic stroke / transient ischemic attack. The patient presented with sudden onset left-sided facial droop and weakness in the left upper extremity."
            course_details = f"Head CT showed no hemorrhage. The patient was evaluated for thrombolysis but was outside the window. Started on dual antiplatelet therapy (Aspirin + Plavix) and high-intensity Atorvastatin. MRI confirmed small acute infarct in the right MCA territory. Physical and occupational therapy were consulted."
            plan_details = f"Aspirin 81mg daily and Clopidogrel 75mg daily. Atorvastatin 80mg daily. Outpatient physical and occupational therapy sessions three times a week. Strict blood pressure control and lifestyle modification."

        summary_text = f"""PATIENT CLINICAL DISCHARGE SUMMARY
=========================================
Patient ID: {pid}
Demographics: {age}-year-old {gender}
Primary Diagnosis: {dx}
Length of Stay: {los} days
Discharge Disposition: {disp}
Admit Source: {row["AdmitSource"]}
=========================================

CHIEF COMPLAINT AND PRESENTATION:
{pronoun_sub} was admitted via the {row["AdmitSource"]} for {dx_details}

PAST MEDICAL HISTORY & COMORBIDITIES:
Number of documented comorbidities: {comorb}.
In addition to the primary admission diagnosis, the patient's history is significant for multiple chronic health conditions including cardiovascular risk factors, metabolic anomalies, and previous healthcare utilization. The patient had {priors} prior hospital admission(s) within the last 12 months.

HOSPITAL COURSE:
{course_details}
Lab testing upon admission and during the stay revealed the following values:
- Hemoglobin A1c (HbA1c): {hba1c}%
- Serum Creatinine: {creat} mg/dL (Baseline normal is 0.6 - 1.2 mg/dL)
- Serum Sodium: {sod} mEq/L (Normal range: 135 - 145 mEq/L)

DISCHARGE STATUS & INSTRUCTIONS:
The patient's condition has stabilized, and {pronoun_sub.lower()} is cleared for discharge to {disp}.
{plan_details}
{pronoun_sub} was instructed on compliance with medications, warning signs of worsening condition (e.g., rapid weight gain, shortness of breath, severe headache), and the necessity of attending all follow-up appointments.

PREVENTATIVE CLINICAL NOTE ON READMISSION RISK:
Given the patient's age ({age}), lab profiles (Creatinine: {creat} mg/dL, HbA1c: {hba1c}%), {priors} prior admissions, and {comorb} comorbidities, the machine learning predictive risk assessment indicates a {'high' if row["ReadmitProbability"] > 0.4 else 'moderate' if row["ReadmitProbability"] > 0.15 else 'low'} probability of readmission. Clinicians should monitor the patient's post-discharge compliance and ensure home care visits are active.
"""
        with open(f"{summaries_dir}/{pid}_summary.txt", "w") as f:
            f.write(summary_text)
            
    print(f"Generated {len(df)} discharge summaries in {summaries_dir}")

if __name__ == "__main__":
    df = generate_synthetic_data()
    generate_discharge_summaries(df)
