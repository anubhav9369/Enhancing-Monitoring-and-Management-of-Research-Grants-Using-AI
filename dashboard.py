import streamlit as st
import joblib
import numpy as np
import json
import os
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

# ---- Initialize session state ----
if 'show_project_setup' not in st.session_state:
    st.session_state.show_project_setup = False
if 'proposal_details' not in st.session_state:
    st.session_state.proposal_details = {}
if 'last_suggested_abstract' not in st.session_state:
    st.session_state.last_suggested_abstract = ""
if 'suggested_abstract' not in st.session_state:
    st.session_state.suggested_abstract = ""

# ---- Database setup ----
def init_db():
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    
    # Projects table — with researcher_name and institution columns guaranteed
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT,
            researcher_name TEXT DEFAULT '',
            institution TEXT DEFAULT '',
            total_budget REAL DEFAULT 0,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'planning',
            predicted_acceptance REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add missing columns if DB already existed without them (migration)
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(projects)").fetchall()]
    for col, default in [("researcher_name", "''"), ("institution", "''"), ("predicted_acceptance", "0")]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE projects ADD COLUMN {col} TEXT DEFAULT {default}")
    
    # Project phases table
    c.execute('''
        CREATE TABLE IF NOT EXISTS project_phases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            phase_name TEXT NOT NULL,
            allocated_budget REAL DEFAULT 0,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'planned',
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')
    
    # Project updates table
    c.execute('''
        CREATE TABLE IF NOT EXISTS project_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            phase_id INTEGER,
            update_text TEXT,
            progress_percentage INTEGER DEFAULT 0,
            budget_used REAL DEFAULT 0,
            update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (phase_id) REFERENCES project_phases (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# ---- Load model + vectorizer ----
@st.cache_resource
def load_resources():
    model = joblib.load("models/grant_acceptance_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    
    feature_names_path = "models/feature_names.json"
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
    else:
        feature_names = None
    
    return model, vectorizer, feature_names

model, vectorizer, feature_names = load_resources()

# ---- Database functions ----
def add_project(title, abstract, researcher_name, institution, total_budget, start_date, end_date, predicted_acceptance):
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO projects (title, abstract, researcher_name, institution, total_budget, start_date, end_date, predicted_acceptance) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (title, abstract, str(researcher_name), str(institution), float(total_budget or 0), start_date, end_date, float(predicted_acceptance or 0))
    )
    project_id = c.lastrowid
    conn.commit()
    conn.close()
    return project_id

def add_project_phase(project_id, phase_name, allocated_budget, start_date, end_date):
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO project_phases (project_id, phase_name, allocated_budget, start_date, end_date) VALUES (?, ?, ?, ?, ?)",
        (project_id, phase_name, float(allocated_budget or 0), start_date, end_date)
    )
    phase_id = c.lastrowid
    conn.commit()
    conn.close()
    return phase_id

def add_project_update(project_id, phase_id, update_text, progress_percentage, budget_used):
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO project_updates (project_id, phase_id, update_text, progress_percentage, budget_used) VALUES (?, ?, ?, ?, ?)",
        (project_id, phase_id, update_text, progress_percentage, float(budget_used or 0))
    )
    
    if progress_percentage >= 100:
        c.execute("UPDATE projects SET status = 'completed' WHERE id = ?", (project_id,))
    elif progress_percentage > 0:
        c.execute("UPDATE projects SET status = 'active' WHERE id = ?", (project_id,))
    
    conn.commit()
    conn.close()

def get_projects():
    conn = sqlite3.connect('grants.db')
    try:
        projects = pd.read_sql("SELECT * FROM projects", conn)
    except Exception:
        projects = pd.DataFrame()
    conn.close()
    return projects

def get_project_phases(project_id):
    conn = sqlite3.connect('grants.db')
    try:
        phases = pd.read_sql("SELECT * FROM project_phases WHERE project_id = ?", conn, params=(project_id,))
    except Exception:
        phases = pd.DataFrame()
    conn.close()
    return phases

def get_project_updates(project_id):
    conn = sqlite3.connect('grants.db')
    try:
        updates = pd.read_sql(
            "SELECT pu.*, pp.phase_name FROM project_updates pu JOIN project_phases pp ON pu.phase_id = pp.id WHERE pu.project_id = ? ORDER BY pu.update_date DESC",
            conn, params=(project_id,)
        )
    except Exception:
        updates = pd.DataFrame()
    conn.close()
    return updates

def get_budget_utilization(project_id):
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    
    c.execute("SELECT total_budget FROM projects WHERE id = ?", (project_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {'total_budget': 0, 'total_used': 0, 'remaining': 0, 'utilization_percent': 0, 'by_phase': pd.DataFrame()}
    total_budget = float(row[0] or 0)
    
    c.execute("SELECT SUM(budget_used) FROM project_updates WHERE project_id = ?", (project_id,))
    total_used = float(c.fetchone()[0] or 0)
    
    try:
        phases = pd.read_sql(
            "SELECT pp.phase_name, pp.allocated_budget, COALESCE(SUM(pu.budget_used), 0) as used FROM project_phases pp LEFT JOIN project_updates pu ON pp.id = pu.phase_id WHERE pp.project_id = ? GROUP BY pp.id",
            conn, params=(project_id,)
        )
    except Exception:
        phases = pd.DataFrame()
    
    conn.close()
    utilization_percent = (total_used / total_budget * 100) if total_budget > 0 else 0
    return {
        'total_budget': total_budget,
        'total_used': total_used,
        'remaining': total_budget - total_used,
        'utilization_percent': utilization_percent,
        'by_phase': phases
    }

def detect_project_issues(project_id):
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    c.execute("SELECT start_date, end_date, status FROM projects WHERE id = ?", (project_id,))
    project = c.fetchone()
    
    if not project:
        conn.close()
        return []
    
    start_date, end_date, status = project
    issues = []
    
    if status == 'active':
        today = datetime.now().date()
        try:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            if today > end_date_obj:
                issues.append({'type': 'schedule', 'severity': 'high', 'message': 'Project is past its end date'})
            elif (end_date_obj - today).days < 30:
                issues.append({'type': 'schedule', 'severity': 'medium', 'message': 'Project is approaching its end date'})
        except Exception:
            pass
    
    budget_info = get_budget_utilization(project_id)
    if budget_info['utilization_percent'] > 100:
        issues.append({'type': 'budget', 'severity': 'high', 'message': f'Budget overused by {budget_info["utilization_percent"] - 100:.1f}%'})
    elif budget_info['utilization_percent'] > 90:
        issues.append({'type': 'budget', 'severity': 'medium', 'message': f'Budget utilization is high ({budget_info["utilization_percent"]:.1f}%)'})
    
    c.execute("SELECT MAX(update_date) FROM project_updates WHERE project_id = ?", (project_id,))
    last_update = c.fetchone()[0]
    if last_update:
        try:
            last_update_date = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
            days_since_update = (datetime.now() - last_update_date).days
            if days_since_update > 30:
                issues.append({'type': 'update', 'severity': 'medium', 'message': f'No updates in {days_since_update} days'})
        except Exception:
            pass
    
    conn.close()
    return issues

# ---- Initialize database ----
init_db()

# ---- Utility functions ----
def calculate_abstract_quality(text):
    quality_indicators = [
        "collaborative", "research", "theory", "supported", "funded",
        "conference", "division", "need", "2025", "innovative",
        "novel", "breakthrough", "advanced", "cutting-edge",
        "impact", "significant", "transformative", "methodology",
        "rigorous", "systematic", "comprehensive", "validated", "framework",
        "approach", "strategy"
    ]
    text_lower = (text or "").lower()
    quality_score = sum(1 for ind in quality_indicators if ind in text_lower)
    
    methodology_indicators = [r"phase \d", r"step \d", r"first.*second.*third", r"methodology", r"approach", r"framework", r"strategy", r"aims?"]
    for pattern in methodology_indicators:
        if re.search(pattern, text_lower):
            quality_score += 2
    
    for ind in ["collaborative", "collaboration", "team", "partnership"]:
        if ind in text_lower:
            quality_score += 2
    
    return min(quality_score / 40, 1.0)

def calculate_funding_appropriateness(amount, abstract_length):
    if amount < 100000:
        return 0.3
    elif amount < 300000:
        return 0.7
    elif amount < 800000:
        return 1.0
    elif amount < 1500000:
        return 0.8
    else:
        return 0.5

def get_missing_keywords(text, important_keywords):
    text_lower = (text or "").lower()
    return [kw for kw in important_keywords if kw not in text_lower]

def generate_suggested_abstract(original_abstract, curr_title, missing_keywords, include_publication_note):
    if not original_abstract:
        original_abstract = ""
    pieces = [
        f"{curr_title}. {original_abstract.strip()}",
        "\n\nProblem & Motivation:\nThis proposal addresses a clear problem with direct scientific and societal relevance. We identify the gap in the literature and the potential impact.",
        "\n\nMethodology & Workplan:\nThe project is organized into clear phases: (1) design and development, (2) validation and experiments, (3) evaluation and dissemination. Each phase contains defined tasks, deliverables, and timelines using rigorous experimental design and appropriate statistical methods.",
        "\n\nExpected Outcomes & Impact:\nWe expect novel validated contributions with demonstrable impact on relevant stakeholders and pathways to dissemination (conference/journal publications, open-source tools, workshops).",
    ]
    if missing_keywords:
        pieces.append(f"\n\nKeywords / Focus Areas: {', '.join(missing_keywords[:7])}.")
    if include_publication_note:
        pieces.append("\n\nTeam & Track Record:\nThe team has prior publications and experience in closely related topics providing the necessary expertise to deliver this work.")
    else:
        pieces.append("\n\nTeam & Support:\nWe will emphasize collaborations, institutional facilities, and advisory support to strengthen the project execution plan.")
    
    suggested = "\n".join(pieces)
    st.session_state.last_suggested_abstract = suggested
    return suggested

# ---- Safe column accessor ----
def safe_get(row, col, default=''):
    """Safely get a value from a DataFrame row, returning default if column missing."""
    try:
        val = row[col]
        return val if pd.notna(val) else default
    except (KeyError, TypeError):
        return default

# ---- Page configuration ----
st.set_page_config(page_title="AI Grant Management System", layout="wide")
st.title("🚀 AI Grant Management System")

# ---- Sidebar navigation ----
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Predict Acceptance", "Project Setup", "Project Tracking", "Reports"])

# ==============================================================================
# PAGE: Predict Acceptance
# ==============================================================================
if page == "Predict Acceptance":
    st.subheader("Grant Acceptance Prediction")
    
    pre = st.session_state.get('proposal_details', {})
    
    with st.form("prediction_form"):
        researcher_name = st.text_input("Researcher Name", value=pre.get('researcher_name', ''))
        institution = st.text_input("Institution", value=pre.get('institution', ''))
        previous_publications = st.selectbox("Previous Publications", ["No", "Yes"], index=1 if pre.get('previous_publications', '') == "Yes" else 0)
        successful_grants = st.number_input("Number of Previously Successful Grants", min_value=0, max_value=100, step=1, value=int(pre.get('successful_grants', 0)))
        prev_title = st.text_input("Title of Previous Research (optional)", value=pre.get('prev_title', ''))
        curr_title = st.text_input("Title of Current Research Proposal", value=pre.get('title', ''))
        abstract = st.text_area("Abstract of Current Proposal", value=pre.get('abstract', ''), height=250)
        funding_amount = st.number_input("Requested Funding Amount", min_value=0.0, step=1000.0, value=float(pre.get('funding_amount', 0.0)))
        submitted = st.form_submit_button("Predict Acceptance")
        
        if submitted:
            if not curr_title.strip():
                st.warning("Please enter a title for the proposal.")
            else:
                combined_text = f"{curr_title}. {abstract}. {prev_title}"
                X_text = vectorizer.transform([combined_text])
                
                if feature_names and 'numerical_features' in feature_names:
                    numerical_features = feature_names['numerical_features']
                    numerical_vector = np.zeros((1, len(numerical_features)))
                    if 'duration_days' in numerical_features:
                        numerical_vector[0, numerical_features.index('duration_days')] = 365
                    X = np.hstack((X_text.toarray(), numerical_vector))
                else:
                    X = X_text
                
                try:
                    text_prob = model.predict_proba(X)[0][1]
                except Exception:
                    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                    text_prob = model.predict_proba(X_arr)[0][1]
                
                abstract_quality = calculate_abstract_quality(combined_text)
                funding_appropriateness = calculate_funding_appropriateness(funding_amount, len(abstract))
                has_pubs = (previous_publications == "Yes")
                
                boost = 0.0
                if has_pubs:
                    boost += 0.15
                if successful_grants > 0:
                    boost += 0.1 * (1 - np.exp(-0.3 * successful_grants))
                boost += 0.2 * abstract_quality
                boost += 0.1 * funding_appropriateness
                
                final_prob = max(0.0, min(1.0, text_prob + boost))
                
                st.write("### Acceptance Prediction")
                
                with st.expander("Prediction Details"):
                    st.write(f"Base Model Score: {text_prob*100:.2f}%")
                    st.write(f"Publications: {'Yes' if has_pubs else 'No'}")
                    st.write(f"Successful Grants: {successful_grants}")
                    st.write(f"Abstract Quality Score: {abstract_quality*100:.1f}%")
                    st.write(f"Funding Appropriateness: {funding_appropriateness*100:.1f}%")
                    st.write(f"Total Boost: {boost*100:.2f}%")
                    st.write(f"Final Probability: {final_prob*100:.2f}%")
                
                if final_prob >= 0.8:
                    st.success(f"🌟 VERY HIGH chance of acceptance — {final_prob*100:.2f}%")
                elif final_prob >= 0.65:
                    st.success(f"✅ HIGH chance of acceptance — {final_prob*100:.2f}%")
                elif final_prob >= 0.4:
                    st.info(f"ℹ️ MODERATE chance of acceptance — {final_prob*100:.2f}%")
                else:
                    st.error(f"❌ LOW chance of acceptance — {final_prob*100:.2f}%")
                
                # Contribution breakdown chart
                contributions_pct = {
                    'Base Model (text)': text_prob * 100,
                    'Publications Boost': 15.0 if has_pubs else 0.0,
                    'Successful Grants Boost': (0.1 * (1 - np.exp(-0.3 * successful_grants)) * 100) if successful_grants > 0 else 0.0,
                    'Abstract Quality Boost': 0.2 * abstract_quality * 100,
                    'Funding Appropriateness Boost': 0.1 * funding_appropriateness * 100
                }
                
                st.write("---")
                st.subheader("Model Contribution Breakdown")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(contributions_pct.keys()),
                    y=list(contributions_pct.values()),
                    text=[f"{v:.1f}%" for v in contributions_pct.values()],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Base model score and metadata boosts (percent points)",
                    yaxis_title="Percent points",
                    xaxis_tickangle=-45,
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store for project setup
                st.session_state.proposal_details = {
                    'title': curr_title,
                    'abstract': abstract,
                    'researcher_name': researcher_name,
                    'institution': institution,
                    'funding_amount': funding_amount,
                    'predicted_acceptance': final_prob,
                    'previous_publications': previous_publications,
                    'successful_grants': successful_grants,
                    'prev_title': prev_title
                }
                
                if final_prob >= 0.65:
                    st.write("---")
                    st.subheader("Next Steps")
                    st.success("Your proposal has a high chance of acceptance! Head to **Project Setup** in the sidebar to configure your project.")
                    st.session_state.show_project_setup = True
                
                if final_prob < 0.40:
                    st.write("---")
                    st.subheader("🔍 Suggestions to Improve Your Proposal")
                    
                    important_keywords = [
                        "innovative", "novel", "collaborative", "methodology", "impact",
                        "framework", "validated", "rigorous", "systematic", "transformative",
                        "approach", "strategy", "evaluation", "dissemination", "benchmarks"
                    ]
                    missing_keywords = get_missing_keywords(combined_text, important_keywords)
                    
                    improvement_points = []
                    if abstract_quality < 0.4:
                        improvement_points.append("• **Improve abstract structure** — Add clearer methodology, step-by-step plan or phases, and explicit objectives and deliverables.")
                    if missing_keywords:
                        improvement_points.append(f"• **Add strong scientific keywords** currently missing: {', '.join(missing_keywords[:7])}.")
                    if not has_pubs:
                        improvement_points.append("• **No publications noted** — reference prior papers or emphasize team expertise and collaborations.")
                    if funding_appropriateness < 0.6:
                        improvement_points.append("• **Funding requested may be misaligned** — justify budget line-items or adjust the amount.")
                    if successful_grants == 0:
                        improvement_points.append("• **No prior successful grants** — highlight institutional support, mentorship, or pilot data.")
                    if len(abstract.strip()) < 120:
                        improvement_points.append("• **Abstract is too short** — expand motivation, methodology, and expected outcomes (aim 150–300 words).")
                    method_patterns = [r"methodolog", r"approach", r"phase", r"experiment", r"evaluat", r"validat", r"benchmark"]
                    if not any(re.search(p, combined_text.lower()) for p in method_patterns):
                        improvement_points.append("• **Methodology details lacking** — specify experiments, datasets, evaluation metrics and success criteria.")
                    
                    for p in improvement_points:
                        st.info(p)
                    
                    if not improvement_points:
                        st.info("Refine your abstract and methodology for stronger justification.")
                    
                    st.write("#### Quick Wins:")
                    quick_wins = []
                    if missing_keywords:
                        quick_wins.append(f"Insert keywords: {', '.join(missing_keywords[:5])}")
                    if len(abstract.strip()) < 120:
                        quick_wins.append("Expand abstract to include methodology and expected outcomes")
                    if not has_pubs:
                        quick_wins.append("Add team/mentorship details or cite prior relevant work")
                    if funding_appropriateness < 0.6:
                        quick_wins.append("Add budget justification or adjust requested amount")
                    for q in quick_wins:
                        st.write(f"- {q}")
                    
                    st.write("---")
                    st.subheader("✍️ Suggested Abstract (editable)")
                    suggested = generate_suggested_abstract(abstract, curr_title, missing_keywords, has_pubs)
                    user_edit = st.text_area("Edit as needed", value=suggested, height=360, key="suggested_text_area")
                    
                    st.session_state['suggested_abstract'] = user_edit
                    st.session_state['last_title'] = curr_title
                    st.session_state['last_researcher'] = researcher_name
                    st.session_state['last_institution'] = institution
                    st.session_state['last_funding'] = funding_amount
                    st.session_state['last_prob'] = final_prob
    
    # Outside-form utilities
    st.write("---")
    st.write("### Suggestion Utilities")
    if st.session_state.get('suggested_abstract'):
        st.markdown("A suggested abstract is available from your last prediction run.")
        if st.button("Use Suggested Abstract for New Prediction"):
            st.session_state.proposal_details = {
                'title': st.session_state.get('last_title', ''),
                'abstract': st.session_state.get('suggested_abstract', ''),
                'researcher_name': st.session_state.get('last_researcher', ''),
                'institution': st.session_state.get('last_institution', ''),
                'funding_amount': st.session_state.get('last_funding', 0.0),
                'predicted_acceptance': st.session_state.get('last_prob', 0)
            }
            st.success("Suggested abstract loaded into the form above.")
            st.rerun()
    else:
        st.write("Run a prediction to generate tailored suggestions and a suggested abstract.")

# ==============================================================================
# PAGE: Project Setup — FIXED
# ==============================================================================
elif page == "Project Setup":
    st.subheader("Project Setup")
    
    # ---- Create New Project Form ----
    st.write("### Create New Project")
    
    pre = st.session_state.get('proposal_details', {})
    
    with st.form("project_setup_form"):
        col1, col2 = st.columns(2)
        project_title = col1.text_input("Project Title", value=pre.get('title', ''))
        researcher_name = col2.text_input("Researcher Name", value=pre.get('researcher_name', ''))
        institution = col1.text_input("Institution", value=pre.get('institution', ''))
        total_budget = col2.number_input("Total Budget ($)", min_value=0.0, step=1000.0, value=float(pre.get('funding_amount', 0.0)))
        
        project_abstract = st.text_area("Project Abstract", value=pre.get('abstract', ''), height=120)
        
        col3, col4 = st.columns(2)
        start_date = col3.date_input("Project Start Date", min_value=datetime.now().date())
        end_date = col4.date_input("Project End Date", value=datetime.now().date() + timedelta(days=365))
        
        st.write("#### Project Phases")
        st.info("Define phases below. The sum of phase budgets should equal the total budget.")
        num_phases = st.number_input("Number of Phases", min_value=1, max_value=10, value=3, step=1)
        
        phases_data = []
        for i in range(int(num_phases)):
            st.write(f"**Phase {i+1}**")
            pc1, pc2, pc3 = st.columns(3)
            phase_name = pc1.text_input(f"Name", value=f"Phase {i+1}", key=f"pname_{i}")
            phase_budget = pc2.number_input(f"Budget ($)", min_value=0.0, step=1000.0, key=f"pbudget_{i}")
            phase_duration = pc3.number_input(f"Duration (days)", min_value=1, value=90, key=f"pdur_{i}")
            phases_data.append({'name': phase_name, 'budget': phase_budget, 'duration': phase_duration})
        
        setup_submitted = st.form_submit_button("✅ Create Project")
        
        if setup_submitted:
            if not project_title.strip():
                st.error("Project title is required.")
            elif end_date <= start_date:
                st.error("End date must be after start date.")
            else:
                # Check budget allocation (warn but don't block if phases are 0)
                total_phase_budget = sum(p['budget'] for p in phases_data)
                if total_budget > 0 and total_phase_budget > 0 and abs(total_phase_budget - total_budget) > 1:
                    st.warning(f"⚠️ Phase budgets total ${total_phase_budget:,.2f} but project total is ${total_budget:,.2f}. Consider adjusting.")
                
                # Create project
                project_id = add_project(
                    project_title, project_abstract, researcher_name, institution,
                    total_budget, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                    float(pre.get('predicted_acceptance', 0))
                )
                
                # Create phases
                phase_start = start_date
                for ph in phases_data:
                    phase_end = phase_start + timedelta(days=int(ph['duration']) - 1)
                    add_project_phase(project_id, ph['name'], ph['budget'],
                                      phase_start.strftime("%Y-%m-%d"), phase_end.strftime("%Y-%m-%d"))
                    phase_start = phase_end + timedelta(days=1)
                
                st.success(f"✅ Project created successfully! Project ID: {project_id}")
                st.info("Go to **Project Tracking** to log progress updates.")
                st.session_state.show_project_setup = False
                st.session_state.proposal_details = {}
                st.rerun()
    
    # ---- Existing Projects List ----
    st.write("---")
    st.write("### Existing Projects")
    projects = get_projects()
    
    if projects.empty:
        st.info("No projects yet. Use the form above to create one.")
    else:
        for _, project in projects.iterrows():
            pid = safe_get(project, 'id', '?')
            ptitle = safe_get(project, 'title', 'Untitled')
            with st.expander(f"Project {pid}: {ptitle}"):
                c1, c2 = st.columns(2)
                c1.write(f"**Researcher:** {safe_get(project, 'researcher_name', 'N/A')}")
                c1.write(f"**Institution:** {safe_get(project, 'institution', 'N/A')}")
                c1.write(f"**Budget:** ${float(safe_get(project, 'total_budget', 0) or 0):,.2f}")
                c1.write(f"**Status:** {safe_get(project, 'status', 'unknown')}")
                c1.write(f"**Dates:** {safe_get(project, 'start_date', '?')} to {safe_get(project, 'end_date', '?')}")
                
                pred_acc = safe_get(project, 'predicted_acceptance', 0)
                try:
                    c2.write(f"**Acceptance Probability:** {float(pred_acc)*100:.2f}%")
                except (ValueError, TypeError):
                    c2.write(f"**Acceptance Probability:** N/A")
                
                phases = get_project_phases(int(pid)) if pid != '?' else pd.DataFrame()
                if not phases.empty:
                    c2.write("**Project Phases:**")
                    for _, phase in phases.iterrows():
                        c2.write(f"- {safe_get(phase, 'phase_name', '?')}: ${float(safe_get(phase, 'allocated_budget', 0) or 0):,.2f}")

# ==============================================================================
# PAGE: Project Tracking — FIXED
# ==============================================================================
elif page == "Project Tracking":
    st.subheader("Project Tracking")
    projects = get_projects()
    
    if projects.empty:
        st.info("No projects found. Create one in **Project Setup**.")
    else:
        project_options = {f"[{safe_get(r, 'id')}] {safe_get(r, 'title', 'Untitled')}": safe_get(r, 'id') for _, r in projects.iterrows()}
        selected_label = st.selectbox("Select a project to track", list(project_options.keys()))
        project_id = project_options[selected_label]
        
        if project_id:
            project = projects[projects['id'] == project_id].iloc[0]
            st.write(f"### Tracking: {safe_get(project, 'title', 'Untitled')}")
            
            col1, col2 = st.columns(2)
            col1.write(f"**Researcher:** {safe_get(project, 'researcher_name', 'N/A')}")
            col1.write(f"**Institution:** {safe_get(project, 'institution', 'N/A')}")
            col1.write(f"**Total Budget:** ${float(safe_get(project, 'total_budget', 0) or 0):,.2f}")
            col1.write(f"**Status:** {safe_get(project, 'status', 'unknown')}")
            col1.write(f"**Dates:** {safe_get(project, 'start_date', '?')} to {safe_get(project, 'end_date', '?')}")
            
            budget_info = get_budget_utilization(project_id)
            col2.write(f"**Budget Used:** ${budget_info['total_used']:,.2f}")
            col2.write(f"**Budget Remaining:** ${budget_info['remaining']:,.2f}")
            col2.write(f"**Utilization:** {budget_info['utilization_percent']:.1f}%")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=budget_info['utilization_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Budget Utilization %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 90], 'color': "gray"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}
                }
            ))
            col2.plotly_chart(fig, use_container_width=True)
            
            issues = detect_project_issues(project_id)
            if issues:
                st.write("### ⚠️ Project Issues")
                for issue in issues:
                    if issue['severity'] == 'high':
                        st.error(f"**{issue['type'].title()}:** {issue['message']}")
                    else:
                        st.warning(f"**{issue['type'].title()}:** {issue['message']}")
            
            # Progress update form
            st.write("### Add Progress Update")
            phases = get_project_phases(project_id)
            
            if phases.empty:
                st.info("No phases defined. Add phases in **Project Setup**.")
            else:
                with st.form("progress_form"):
                    phase_options = {}
                    for _, phase in phases.iterrows():
                        label = f"{safe_get(phase, 'phase_name', '?')} (${float(safe_get(phase, 'allocated_budget', 0) or 0):,.2f})"
                        phase_options[label] = safe_get(phase, 'id')
                    
                    selected_phase = st.selectbox("Select Phase", list(phase_options.keys()))
                    phase_id = phase_options[selected_phase]
                    
                    update_text = st.text_area("Update Description", height=150)
                    progress_percentage = st.slider("Progress Percentage", 0, 100, 0)
                    
                    # Calculate remaining budget for this phase
                    phase_updates = get_project_updates(project_id)
                    phase_budget_used = 0.0
                    if not phase_updates.empty and 'phase_id' in phase_updates.columns:
                        used_in_phase = phase_updates[phase_updates['phase_id'] == phase_id]
                        if not used_in_phase.empty:
                            phase_budget_used = float(used_in_phase['budget_used'].sum() or 0)
                    
                    phase_row = phases[phases['id'] == phase_id]
                    allocated = float(phase_row.iloc[0]['allocated_budget'] or 0) if not phase_row.empty else 0.0
                    remaining_in_phase = max(0.0, allocated - phase_budget_used)
                    
                    budget_used = st.number_input(
                        "Budget Used in This Update ($)",
                        min_value=0.0,
                        max_value=remaining_in_phase if remaining_in_phase > 0 else 999999999.0,
                        step=100.0,
                        value=0.0
                    )
                    
                    if st.form_submit_button("Submit Update"):
                        if not update_text.strip():
                            st.warning("Please provide an update description.")
                        else:
                            add_project_update(project_id, phase_id, update_text, progress_percentage, budget_used)
                            st.success("✅ Update added successfully!")
                            st.rerun()
            
            # Progress history
            st.write("### Progress History")
            updates = get_project_updates(project_id)
            
            if updates.empty:
                st.info("No updates yet.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=updates['update_date'],
                    y=updates['progress_percentage'],
                    mode='lines+markers',
                    name='Progress %',
                    line=dict(width=3)
                ))
                fig.update_layout(
                    title='Project Progress Over Time',
                    xaxis_title='Date',
                    yaxis_title='Progress (%)',
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                budget_by_phase = budget_info['by_phase']
                if not budget_by_phase.empty:
                    fig = px.bar(
                        budget_by_phase,
                        x='phase_name',
                        y=['allocated_budget', 'used'],
                        title='Budget Allocation vs. Usage by Phase',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                for _, update in updates.iterrows():
                    with st.expander(f"Update on {safe_get(update, 'update_date', '?')} — {safe_get(update, 'phase_name', '?')}"):
                        st.write(f"**Progress:** {safe_get(update, 'progress_percentage', 0)}%")
                        st.write(f"**Budget Used:** ${float(safe_get(update, 'budget_used', 0) or 0):,.2f}")
                        st.write(safe_get(update, 'update_text', ''))

# ==============================================================================
# PAGE: Reports
# ==============================================================================
elif page == "Reports":
    st.subheader("Grant Management Reports")
    projects = get_projects()
    
    if projects.empty:
        st.info("No projects found. Create a project in **Project Setup**.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Projects", len(projects))
        col2.metric("Active Projects", len(projects[projects['status'] == 'active']))
        col3.metric("Completed Projects", len(projects[projects['status'] == 'completed']))
        
        total_budget = float(projects['total_budget'].sum() or 0)
        total_used = sum(get_budget_utilization(int(pid))['total_used'] for pid in projects['id'])
        col4.metric("Total Budget Utilization", f"{(total_used/total_budget*100):.1f}%" if total_budget > 0 else "0.0%")
        
        st.write("### Project Status Distribution")
        status_counts = projects['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.pie(status_counts, values='Count', names='Status', title="Project Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Budget Utilization by Project")
        budget_data = []
        for _, project in projects.iterrows():
            budget_info = get_budget_utilization(int(safe_get(project, 'id', 0)))
            budget_data.append({
                'Project': safe_get(project, 'title', 'Untitled'),
                'Budget Utilization %': budget_info['utilization_percent'],
                'Status': safe_get(project, 'status', 'unknown')
            })
        
        budget_df = pd.DataFrame(budget_data)
        fig = px.bar(budget_df, x='Project', y='Budget Utilization %', color='Status', title="Budget Utilization by Project")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Project Issues Report")
        all_issues = []
        for project_id in projects['id']:
            issues = detect_project_issues(int(project_id))
            for issue in issues:
                ptitle = projects[projects['id'] == project_id].iloc[0]
                all_issues.append({
                    'Project': safe_get(ptitle, 'title', 'Untitled'),
                    'Type': issue['type'],
                    'Severity': issue['severity'],
                    'Message': issue['message']
                })
        
        if all_issues:
            issues_df = pd.DataFrame(all_issues)
            st.dataframe(issues_df)
            issues_by_type = issues_df.groupby('Type').size().reset_index(name='Count')
            fig = px.bar(issues_by_type, x='Type', y='Count', title="Issues by Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No issues detected across all projects!")
