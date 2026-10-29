import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from src.data.synthetic_generator import generate_base_telemetry, inject_anomalies
from src.features.build_features import extract_rolling_features

def train_isolation_forest(contamination: float = 0.05, random_state: int = 42):
    print("Generating telemetry and extracting rolling features...")
    raw_df = generate_base_telemetry(num_samples=10000)
    fault_df = inject_anomalies(raw_df, spike_ratio=contamination, block_size=50)
    featured_df = extract_rolling_features(fault_df, window_size=50)
    
    feature_cols = [col for col in featured_df.columns if 'roll' in col]
    X = featured_df[feature_cols]
    y_true = featured_df['label']
    
    print(f"Training Isolation Forest model (contamination={contamination})...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)
    
    raw_preds = model.predict(X)
    y_pred = np.where(raw_preds == -1, 1, 0)
    
    print("\n" + "="*45)
    print("   ISOLATION FOREST MODEL PERFORMANCE REPORT   ")
    print("="*45)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Anomaly (1)']))
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*45)
    
    # Save trained model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/isolation_forest.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel artifact successfully saved to: {model_path}")
    
    return model, featured_df

if __name__ == "__main__":
    train_isolation_forest()