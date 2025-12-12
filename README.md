# 📘 AI-Based Research Grant Monitoring & Management System

An end-to-end AI + Streamlit platform designed to automate the evaluation, monitoring, and progress tracking of research grant projects.
This system combines NLP (SciBERT), ML models, automated alerts, project tracking, and interactive dashboards to support funding agencies, universities, and research organizations.

#Demo

<img width="1050" height="896" alt="image" src="https://github.com/user-attachments/assets/9283408d-c99a-4d3e-9bf3-3af7c1906be0" />
<img width="828" height="660" alt="image" src="https://github.com/user-attachments/assets/7363872e-45ea-4c70-8637-a8390472dffe" />

# ⭐ Features

🔍 1. Proposal Acceptance Prediction (AI Model)
Uses SciBERT embeddings + XGBoost to predict grant proposal acceptance.
Provides explainable insights (SHAP-based feature importance).

📈 2. Project Progress Tracking
Add project updates (date, description, progress %).
Generates a time-series line chart of progress.
Detects:
Stalled projects
Missing updates
Abnormal delays

💰 3. Budget Utilization & Analytics Dashboard
Phase-wise budget graphs.
Funding allocation insights.
Visualization via Plotly.

📊 4. Smart Reporting Dashboard
Includes:
Number of active projects
Category-wise performance
Funding distribution
Progress insights

🛠 5. Admin Panel Tools
Manage proposals
Add project updates
View analytics & alerts (via Mail)

# 🗂️ Project Structure
├── new_dashboard.py         # Main Streamlit dashboard
├── app.py                   # Optional backend API
├── dataset/
│   ├── grants_db.csv
│   └── awards_full_data.csv
├── models/
│   ├── scibert_model/
│   ├── xgboost_model.pkl
│   └── tfidf_vectorizer.pkl
├── frontend/                # React/HTML assets
│   ├── public/
│   ├── src/
├── requirements.txt
└── README.md


# 🔧 Tech Stack
Frontend / Dashboard
Streamlit
Plotly
HTML/CSS (UI enhancements)
Machine Learning
SciBERT (HuggingFace)
XGBoost
TF-IDF
SHAP for explainability
Backend
Python
Pandas, NumPy


# 🚀 Installation & Setup
1️⃣ Clone the repo
git clone <your-repo-url>
cd project-folder

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit dashboard
streamlit run new_dashboard.py
