# Improved training code with overfitting fixes
import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from datetime import datetime

def read_csv_data(csv_path):
    """Read data from CSV file"""
    print(f"Reading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} records in CSV.")
    return df

def extract_features(df):
    """Extract features from the CSV dataframe"""
    data = []
    
    for _, row in df.iterrows():
        # Extract text features (adjust column names as needed)
        title = str(row.get("awd_titl_txt", row.get("title", "")))
        abstract = str(row.get("awd_abstract_narration", row.get("abstract", "")))
        program_text = str(row.get("awd_program_text", ""))
        
        # Extract numerical features (EXCLUDING funding to prevent leakage)
        # funding = float(row.get("awd_amount", 0))  # Removed to prevent leakage
        
        # Handle dates - adjust column names as needed
        start_date = str(row.get("awd_effective_date", ""))
        end_date = str(row.get("awd_latest_amend_date", ""))
        
        # Calculate duration if dates are available
        duration_days = 0
        if start_date and end_date and start_date != "nan" and end_date != "nan":
            try:
                # Try different date formats
                for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
                    try:
                        start = datetime.strptime(start_date, fmt)
                        end = datetime.strptime(end_date, fmt)
                        duration_days = (end - start).days
                        break
                    except ValueError:
                        continue
            except:
                pass
        
        # Extract categorical features
        award_type = str(row.get("award_type", ""))
        agency = str(row.get("awarding_agency_code", ""))
        
        # PI information - adjust column names as needed
        pi_info = str(row.get("pi_name", row.get("pi_full_name", "")))
        
        # Combine text features
        text = f"{title}. {abstract}. {program_text}. {pi_info}"
        
        data.append({
            "text": text,
            # "funding": funding,  # Removed to prevent leakage
            "duration_days": duration_days,
            "award_type": award_type,
            "agency": agency
        })
    
    df_features = pd.DataFrame(data)
    df_features = df_features[df_features["text"].str.strip() != ""].reset_index(drop=True)
    return df_features

def train(csv_path, out_dir):
    """Train the model using CSV data"""
    # Read CSV data
    df = read_csv_data(csv_path)
    
    # Extract features
    features_df = extract_features(df)
    if features_df.empty:
        raise ValueError("No valid records found with abstracts or titles.")
    
    print(f"Processing {len(features_df)} valid records...")
    
    # Create a more meaningful label - high funding as proxy for successful projects
    funding_threshold = df["awd_amount"].quantile(0.75)  # Top 25% as "successful"
    features_df["label"] = (df["awd_amount"] > funding_threshold).astype(int)
    
    # Handle categorical variables
    features_df = pd.get_dummies(features_df, columns=["award_type", "agency"], drop_first=True)
    
    # Separate text and numerical features
    text_features = features_df["text"]
    numerical_features = features_df.drop(columns=["text", "label"])
    
    print("Vectorizing text...")
    # Reduce max_features to prevent overfitting
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(text_features)
    
    # Combine text and numerical features
    X_numerical = numerical_features.values
    X = np.hstack((X_text.toarray(), X_numerical))
    y = features_df["label"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model with regularization to prevent overfitting...")
    # Use a simpler model with regularization
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced number of trees
        max_depth=10,     # Limit tree depth
        min_samples_split=5,  # Require more samples to split
        min_samples_leaf=2,   # Require more samples at leaf
        max_features='sqrt',  # Consider fewer features at each split
        random_state=42
    )
    
    # Perform cross-validation to get a better estimate of performance
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on test set
    preds = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print("Test ROC AUC:", roc_auc_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Feature importance analysis
    feature_names = vectorizer.get_feature_names_out().tolist() + numerical_features.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 important features:")
    for i in range(min(10, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "grant_acceptance_model.pkl"))
    joblib.dump(vectorizer, os.path.join(out_dir, "tfidf_vectorizer.pkl"))
    
    # Save feature names for later use in prediction
    feature_names_dict = {
        'text_features': vectorizer.get_feature_names_out().tolist(),
        'numerical_features': numerical_features.columns.tolist()
    }
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names_dict, f)
    
    print(f"✅ Model, vectorizer, and feature names saved in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing award data")
    parser.add_argument("--out_dir", required=True, help="Directory to save trained model")
    args = parser.parse_args()
    train(args.csv, args.out_dir)