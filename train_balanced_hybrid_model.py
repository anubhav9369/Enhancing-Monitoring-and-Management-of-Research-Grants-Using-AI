import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1 Load and Clean Dataset
print("\n Loading dataset...")
df = pd.read_csv("/Users/anubhavverma/Desktop/mp/awards_full_data.csv")
print(f" Loaded {len(df)} records (raw)")

# Clean funding column
df["funded"] = df["funded"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
df = df.dropna(subset=["awd_abstract_narration", "awd_amount", "funded"])
print(f" Funded ratio: {df['funded'].value_counts().to_dict()}")

# 2️ Create Balanced Subset
max_samples_per_class = min(df["funded"].value_counts().max(), 1000)
df_balanced = pd.concat([
    df[df["funded"] == 1].sample(min(max_samples_per_class, len(df[df["funded"] == 1])), random_state=42),
    df[df["funded"] == 0].sample(min(max_samples_per_class, len(df[df["funded"] == 0])), random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f" Balanced-ish data used for training: {len(df_balanced)} samples")

# 3 Load SciBERT Model
print(" Loading SciBERT tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
bert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
bert_model.eval()

# 4 Generate Embeddings (Batched)
def get_embeddings_batch(texts, batch_size=8):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating SciBERT embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

print(" Generating SciBERT embeddings (batched)...")
embeddings = get_embeddings_batch(df_balanced["awd_abstract_narration"].tolist())
print(f" Embeddings shape: {embeddings.shape}")

# 5️ Prepare Metadata
print("🔧 Preparing metadata features...")
meta_features = pd.DataFrame({
    "Duration": np.log1p(df_balanced["awd_duration"]),
    "num_publications": np.random.randint(0, 5, len(df_balanced)),
    "num_grants": np.random.randint(0, 3, len(df_balanced)),
})


# 6️ Combine All Features
X = np.hstack([embeddings, meta_features.values])
y = df_balanced["funded"].values

# Scale metadata part only
scaler = StandardScaler()
X[:, -meta_features.shape[1]:] = scaler.fit_transform(X[:, -meta_features.shape[1]:])

print(f" Final X shape: {X.shape}  y shape: {y.shape}")

# 7️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y
)
print(f" Split: {X_train.shape} {X_test.shape}")


# 8️ Model Definition
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)

# 9️ Cross Validation
print("\n Performing 5-Fold Cross-Validation (stability check)...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X, y, cv=kfold, scoring="accuracy")
print(f" CV Accuracy Scores: {cv_scores}")
print(f" Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 10 Train + Calibrate Model
print("\n⚙️ Training XGBoost with sigmoid calibration...")
xgb.fit(X_train, y_train)
calibrated_model = CalibratedClassifierCV(estimator=xgb, cv=3, method='sigmoid')
calibrated_model.fit(X_train, y_train)

train_acc = calibrated_model.score(X_train, y_train)
test_acc = calibrated_model.score(X_test, y_test)
print(f" Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

# 1️1️ SHAP Feature Importance
print("\n Computing SHAP feature importances (top 20)...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test[:100])

shap_importance = np.abs(shap_values).mean(axis=0)
feature_names = [f"feat_{i}" for i in range(X.shape[1])]
importance_df = pd.DataFrame({"feature": feature_names, "importance": shap_importance})
top_features = importance_df.sort_values(by="importance", ascending=False).head(20)

print(top_features)

# 1️2️ Save Final Model
print("\n Saving verified model as funding_hybrid_full_model_verified.pkl ...")
joblib.dump({
    "model": calibrated_model,
    "scaler": scaler,
    "tokenizer": tokenizer,
    "bert_model": bert_model,
    "meta_features": meta_features.columns.tolist()
}, "funding_hybrid_full_model_verified.pkl")

print(" Model bundle saved successfully!")

# 1️3️ Save Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 SHAP Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("\n Feature importance plot saved as feature_importance.png")
