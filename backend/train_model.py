import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate():
    print("Loading patient dataset...")
    csv_path = "data/patients.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_data.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    # Drop PatientID (not a feature) and ReadmitProbability (true probability, would cause leakage)
    X = df.drop(columns=["PatientID", "ReadmitProbability", "Readmitted"])
    y = df["Readmitted"]
    
    # Define features
    categorical_cols = ["Gender", "PrimaryDiagnosis", "DischargeDisposition", "AdmitSource"]
    numerical_cols = ["Age", "BMI", "LengthOfStay", "PriorAdmissions", "Comorbidities", "LabHbA1c", "LabCreatinine", "LabSodium"]
    
    print("Defining preprocessing pipeline...")
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    
    # Define models to compare
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss"
        ),
        "MLP (Neural Network)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_avg_auc = 0
    best_model_name = None
    best_pipeline = None
    
    # Evaluate each model using 5-fold CV
    for name, model in models.items():
        print(f"\nEvaluating {name} using 5-fold cross-validation...")
        accs, aucs, f1s = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            f1 = f1_score(y_val, y_pred)
            
            accs.append(acc)
            aucs.append(auc)
            f1s.append(f1)
            
            print(f"  Fold {fold} - Accuracy: {acc:.3f}, AUC: {auc:.3f}, F1: {f1:.3f}")
            
        avg_acc = np.mean(accs)
        avg_auc = np.mean(aucs)
        avg_f1 = np.mean(f1s)
        
        print(f"{name} CV Summary:")
        print(f"  Average Accuracy: {avg_acc:.4f}")
        print(f"  Average ROC-AUC : {avg_auc:.4f}")
        print(f"  Average F1-score: {avg_f1:.4f}")
        
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_model_name = name
            # Fit best pipeline on full dataset
            best_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])
            best_pipeline.fit(X, y)
            
    print(f"\nBest model selected: {best_model_name} (ROC-AUC: {best_avg_auc:.4f})")
    
    # Save best pipeline model
    os.makedirs("models", exist_ok=True)
    model_path = "models/readmission_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Saved best model pipeline to {model_path}")
    
    # Generate feature importance report if XGBoost/RandomForest was selected
    classifier = best_pipeline.named_steps["classifier"]
    
    # Get feature names from preprocessor
    # Fit preprocessor separately to reconstruct feature names
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    
    # Numerical features remain the same
    num_feature_names = numerical_cols
    # Get categorical encoded names
    cat_encoder = preprocessor_fitted.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    
    all_feature_names = num_feature_names + cat_feature_names
    
    feature_importances = {}
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        # Train an auxiliary XGBoost model to get feature importances for explainability
        print("\nSelected classifier does not support feature_importances_. Training auxiliary XGBoost for explainability...")
        helper_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss"
        )
        X_trans = preprocessor_fitted.transform(X)
        helper_xgb.fit(X_trans, y)
        importances = helper_xgb.feature_importances_
        
    feature_importances = dict(zip(all_feature_names, importances))
    # Sort by importance
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\nGlobal Feature Importances:")
    for feat, val in sorted_importances[:10]:
        print(f"  {feat}: {val:.4f}")
        
    # Save feature importances metadata
    importances_path = "models/feature_importances.pkl"
    with open(importances_path, "wb") as f:
        pickle.dump({
            "feature_names": all_feature_names,
            "importances": feature_importances,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols
        }, f)
    print(f"Saved feature importances to {importances_path}")
        
    return best_model_name, best_avg_auc

if __name__ == "__main__":
    train_and_evaluate()
