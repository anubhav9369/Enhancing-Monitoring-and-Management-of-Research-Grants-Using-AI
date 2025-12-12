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
    
    # Projects table
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT,
            researcher_name TEXT,
            institution TEXT,
            total_budget REAL,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'planning',
            predicted_acceptance REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Project phases table
    c.execute('''
        CREATE TABLE IF NOT EXISTS project_phases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            phase_name TEXT NOT NULL,
            allocated_budget REAL,
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
            progress_percentage INTEGER,
            budget_used REAL,
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
    # adjust paths if needed
    model = joblib.load("models/grant_acceptance_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    
    # Load feature names if available
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
        (title, abstract, researcher_name, institution, total_budget, start_date, end_date, predicted_acceptance)
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
        (project_id, phase_name, allocated_budget, start_date, end_date)
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
        (project_id, phase_id, update_text, progress_percentage, budget_used)
    )
    
    # Update project status based on progress
    if progress_percentage >= 100:
        c.execute("UPDATE projects SET status = 'completed' WHERE id = ?", (project_id,))
    elif progress_percentage > 0:
        c.execute("UPDATE projects SET status = 'active' WHERE id = ?", (project_id,))
    
    conn.commit()
    conn.close()

def get_projects():
    conn = sqlite3.connect('grants.db')
    projects = pd.read_sql("SELECT * FROM projects", conn)
    conn.close()
    return projects

def get_project_phases(project_id):
    conn = sqlite3.connect('grants.db')
    phases = pd.read_sql("SELECT * FROM project_phases WHERE project_id = ?", conn, params=(project_id,))
    conn.close()
    return phases

def get_project_updates(project_id):
    conn = sqlite3.connect('grants.db')
    updates = pd.read_sql(
        "SELECT pu.*, pp.phase_name FROM project_updates pu JOIN project_phases pp ON pu.phase_id = pp.id WHERE pu.project_id = ? ORDER BY pu.update_date DESC",
        conn, params=(project_id,)
    )
    conn.close()
    return updates

def get_budget_utilization(project_id):
    conn = sqlite3.connect('grants.db')
    
    # Get total budget
    c = conn.cursor()
    c.execute("SELECT total_budget FROM projects WHERE id = ?", (project_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {
            'total_budget': 0,
            'total_used': 0,
            'remaining': 0,
            'utilization_percent': 0,
            'by_phase': pd.DataFrame()
        }
    total_budget = row[0] or 0
    
    # Get total budget used
    c.execute("SELECT SUM(budget_used) FROM project_updates WHERE project_id = ?", (project_id,))
    total_used = c.fetchone()[0] or 0
    
    # Get budget by phase
    phases = pd.read_sql(
        "SELECT pp.phase_name, pp.allocated_budget, COALESCE(SUM(pu.budget_used), 0) as used FROM project_phases pp LEFT JOIN project_updates pu ON pp.id = pu.phase_id WHERE pp.project_id = ? GROUP BY pp.id",
        conn, params=(project_id,)
    )
    
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
    
    # Get project details
    c = conn.cursor()
    c.execute("SELECT start_date, end_date, status FROM projects WHERE id = ?", (project_id,))
    project = c.fetchone()
    
    if not project:
        conn.close()
        return []
    
    start_date, end_date, status = project
    issues = []
    
    # Check if project is behind schedule
    if status == 'active':
        today = datetime.now().date()
        try:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            if today > end_date_obj:
                issues.append({
                    'type': 'schedule',
                    'severity': 'high',
                    'message': 'Project is past its end date'
                })
            elif (end_date_obj - today).days < 30:
                issues.append({
                    'type': 'schedule',
                    'severity': 'medium',
                    'message': 'Project is approaching its end date'
                })
        except Exception:
            pass
    
    # Check for budget issues
    budget_info = get_budget_utilization(project_id)
    if budget_info['utilization_percent'] > 100:
        issues.append({
            'type': 'budget',
            'severity': 'high',
            'message': f'Budget overused by {budget_info["utilization_percent"] - 100:.1f}%'
        })
    elif budget_info['utilization_percent'] > 90:
        issues.append({
            'type': 'budget',
            'severity': 'medium',
            'message': f'Budget utilization is high ({budget_info["utilization_percent"]:.1f}%)'
        })
    
    # Check for lack of recent updates
    c.execute("SELECT MAX(update_date) FROM project_updates WHERE project_id = ?", (project_id,))
    last_update = c.fetchone()[0]
    
    if last_update:
        try:
            last_update_date = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
            days_since_update = (datetime.now() - last_update_date).days
            if days_since_update > 30:
                issues.append({
                    'type': 'update',
                    'severity': 'medium',
                    'message': f'No updates in {days_since_update} days'
                })
        except Exception:
            pass
    
    conn.close()
    return issues

# ---- Initialize database ----
init_db()

# ---- Utility functions for explainability + suggestions ----
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
    quality_score = 0
    
    # Count occurrences of quality indicators
    for indicator in quality_indicators:
        if indicator in text_lower:
            quality_score += 1
    
    # Check for structured methodology
    methodology_indicators = [
        r"phase \d", r"step \d", r"first.*second.*third",
        r"methodology", r"approach", r"framework", r"strategy", r"aims?"
    ]
    
    for pattern in methodology_indicators:
        if re.search(pattern, text_lower):
            quality_score += 2
    
    # Check for collaboration mentions
    collaboration_indicators = ["collaborative", "collaboration", "team", "partnership"]
    for indicator in collaboration_indicators:
        if indicator in text_lower:
            quality_score += 2
    
    # Normalize score (max around 40)
    return min(quality_score / 40, 1.0)

def calculate_funding_appropriateness(amount, abstract_length):
    # Base score on funding amount (higher isn't always better)
    if amount < 100000:
        return 0.3  # Too low for most research
    elif amount < 300000:
        return 0.7  # Reasonable for small projects
    elif amount < 800000:
        return 1.0  # Optimal range
    elif amount < 1500000:
        return 0.8  # Large but reasonable
    else:
        return 0.5  # Very large, might be questionable

def get_missing_keywords(text, important_keywords):
    text_lower = (text or "").lower()
    missing = [kw for kw in important_keywords if kw not in text_lower]
    return missing

def generate_suggested_abstract(original_abstract, curr_title, missing_keywords, include_publication_note):
    """
    Create a suggested abstract by adding structure, methodology, expected impact and missing keywords.
    This is a heuristic/template—adjust to taste.
    """
    if not original_abstract:
        original_abstract = ""
    pieces = []
    # Short intro: what and why
    pieces.append(f"{curr_title}. {original_abstract.strip()}")
    pieces.append("\n\nProblem & Motivation:\nThis proposal addresses a clear problem with direct scientific and societal relevance. We motivate the work by identifying the gap in the literature and the potential impact.")
    
    # Methodology: phases + methods
    pieces.append("\n\nMethodology & Workplan:\nThe project is organized into clear phases: (1) design and development, (2) validation and experiments, (3) evaluation and dissemination. Each phase will contain defined tasks, deliverables, and timelines. The methodology employs rigorous experimental design, quantitative evaluation, and appropriate statistical methods.")
    
    # Expected outcomes / impact
    pieces.append("\n\nExpected Outcomes & Impact:\nWe expect novel contributions that are validated experimentally, with demonstrable impact on relevant stakeholders and pathways to dissemination (conference/journal publications, open-source tools, workshops).")
    
    # Add missing keywords as a final 'keywords' sentence
    if missing_keywords:
        mk = ", ".join(missing_keywords[:7])
        pieces.append(f"\n\nKeywords / Focus Areas: {mk}.")
    
    # Optionally mention prior outputs / publications
    if include_publication_note:
        pieces.append("\n\nTeam & Track Record:\nThe team has prior publications and experience in closely related topics which provide the necessary expertise to deliver the proposed work.")
    else:
        pieces.append("\n\nTeam & Support:\nWe will emphasize collaborations, institutional facilities, and advisory support to strengthen the project execution plan.")
    
    suggested = "\n".join(pieces)
    # store for session
    st.session_state.last_suggested_abstract = suggested
    return suggested

# ---- Page configuration ----
st.set_page_config(page_title="AI Grant Management System", layout="wide")
st.title("🚀 AI Grant Management System")

# ---- Sidebar navigation ----
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Predict Acceptance", "Project Setup", "Project Tracking", "Reports"])

if page == "Predict Acceptance":
    st.subheader("Grant Acceptance Prediction")
    
    # Project setup form (shown conditionally)
    if st.session_state.show_project_setup:
        st.write("---")
        st.subheader("Project Setup")
        st.write("Your proposal has a high chance of acceptance! Set up your project details below:")
        
        with st.form("project_setup"):
            # Get proposal details from session state
            proposal = st.session_state.proposal_details
            
            st.write("#### Project Setup")
            project_title = st.text_input("Project Title", value=proposal.get('title', ''))
            project_abstract = st.text_area("Project Abstract", value=proposal.get('abstract', ''), height=150)
            start_date = st.date_input("Project Start Date", min_value=datetime.now().date())
            end_date = st.date_input("Project End Date", min_value=start_date + timedelta(days=30))
            
            st.write("#### Budget Allocation")
            total_budget = st.number_input("Total Budget", min_value=proposal.get('funding_amount', 0), value=proposal.get('funding_amount', 0), step=1000.0)
            
            st.write("#### Project Phases")
            num_phases = st.number_input("Number of Phases", min_value=1, max_value=10, value=3, step=1)
            
            phases = []
            for i in range(int(num_phases)):
                st.write(f"**Phase {i+1}**")
                col1, col2, col3 = st.columns(3)
                phase_name = col1.text_input(f"Phase Name", value=f"Phase {i+1}")
                phase_budget = col2.number_input(f"Budget Allocation", min_value=0.0, step=1000.0)
                phase_duration = col3.number_input(f"Duration (days)", min_value=1, value=90)
                
                # Calculate phase dates
                if i == 0:
                    phase_start = start_date
                else:
                    phase_start = phases[i-1]['end_date'] + timedelta(days=1)
                
                phase_end = phase_start + timedelta(days=int(phase_duration)-1)
                
                phases.append({
                    'name': phase_name,
                    'budget': phase_budget,
                    'start_date': phase_start,
                    'end_date': phase_end
                })
            
            setup_submitted = st.form_submit_button("Create Project")
            
            if setup_submitted:
                # Validate budget allocation
                total_phase_budget = sum(phase['budget'] for phase in phases)
                if abs(total_phase_budget - total_budget) > 0.01:
                    st.error(f"Total phase budget ({total_phase_budget}) must equal total project budget ({total_budget})")
                else:
                    # Create project
                    project_id = add_project(
                        project_title, project_abstract, proposal.get('researcher_name', ''), proposal.get('institution', ''),
                        total_budget, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                        proposal.get('predicted_acceptance', 0)
                    )
                    
                    # Create phases
                    for phase in phases:
                        add_project_phase(
                            project_id, phase['name'], phase['budget'],
                            phase['start_date'].strftime("%Y-%m-%d"), phase['end_date'].strftime("%Y-%m-%d")
                        )
                    
                    st.success(f"Project created successfully! Project ID: {project_id}")
                    st.info("You can now track your project in the 'Project Tracking' page.")
                    
                    # Reset session state
                    st.session_state.show_project_setup = False
                    st.session_state.proposal_details = {}
    
    # Prediction form
    # Prefill values from session_state.proposal_details if present
    pre = st.session_state.get('proposal_details', {})
    with st.form("prediction_form"):
        researcher_name = st.text_input("Researcher Name", value=pre.get('researcher_name', ''))
        institution = st.text_input("Institution", value=pre.get('institution', ''))
        previous_publications = st.selectbox("Previous Publications", ["No", "Yes"], index=1 if pre.get('previous_publications','')=="Yes" else 0)
        successful_grants = st.number_input("Number of Previously Successful Grants", min_value=0, max_value=100, step=1, value=int(pre.get('successful_grants', 0)))
        prev_title = st.text_input("Title of Previous Research (optional)", value=pre.get('prev_title',''))
        curr_title = st.text_input("Title of Current Research Proposal", value=pre.get('title',''))
        abstract = st.text_area("Abstract of Current Proposal", value=pre.get('abstract',''), height=250)
        funding_amount = st.number_input("Requested Funding Amount", min_value=0.0, step=1000.0, value=float(pre.get('funding_amount', 0.0)))
        submitted = st.form_submit_button("Predict Acceptance")
        
        if submitted:
            if not curr_title.strip():
                st.warning("Please enter a title for the proposal.")
            else:
                # Prepare text features
                combined_text = f"{curr_title}. {abstract}. {prev_title}"
                X_text = vectorizer.transform([combined_text])
                
                # Create numerical features to match training data
                if feature_names and 'numerical_features' in feature_names:
                    numerical_features = feature_names['numerical_features']
                    n_numerical = len(numerical_features)
                    
                    # Create a zero vector for numerical features
                    numerical_vector = np.zeros((1, n_numerical))
                    
                    # Set duration_days if it exists in numerical features
                    if 'duration_days' in numerical_features:
                        idx = numerical_features.index('duration_days')
                        numerical_vector[0, idx] = 365  # Default duration of 1 year
                    
                    # Combine text and numerical features
                    X = np.hstack((X_text.toarray(), numerical_vector))
                else:
                    # Fallback: use only text features
                    X = X_text
                
                # Get base prediction from model
                try:
                    text_prob = model.predict_proba(X)[0][1]
                except Exception:
                    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                    text_prob = model.predict_proba(X_arr)[0][1]
                
                # Calculate quality indicators from abstract
                abstract_quality = calculate_abstract_quality(combined_text)
                
                # Calculate funding appropriateness
                funding_appropriateness = calculate_funding_appropriateness(funding_amount, len(abstract))
                
                # Calculate metadata boost
                has_pubs = (previous_publications == "Yes")
                boost = 0.0
                
                # Researcher track record boost
                if has_pubs:
                    boost += 0.15  # Significant boost for publications
                
                # Successful grants boost (diminishing returns)
                if successful_grants > 0:
                    boost += 0.1 * (1 - np.exp(-0.3 * successful_grants))
                
                # Abstract quality boost
                boost += 0.2 * abstract_quality
                
                # Funding appropriateness boost
                boost += 0.1 * funding_appropriateness
                
                # Ensure final probability is between 0 and 1
                final_prob = max(0.0, min(1.0, text_prob + boost))
                
                # Display prediction
                st.write("### Acceptance Prediction")
                
                # Show debug information
                with st.expander("Prediction Details"):
                    st.write(f"Base Model Score: {text_prob*100:.2f}%")
                    st.write(f"Publications: {'Yes' if has_pubs else 'No'}")
                    st.write(f"Successful Grants: {successful_grants}")
                    st.write(f"Abstract Quality Score: {abstract_quality*100:.1f}%")
                    st.write(f"Funding Appropriateness: {funding_appropriateness*100:.1f}%")
                    st.write(f"Total Boost: {boost*100:.2f}%")
                    st.write(f"Final Probability: {final_prob*100:.2f}%")
                
                if final_prob >= 0.8:
                    label = "VERY HIGH chance of acceptance"
                    st.success(f"🌟 {label} — {final_prob*100:.2f}%")
                elif final_prob >= 0.65:
                    label = "HIGH chance of acceptance"
                    st.success(f"✅ {label} — {final_prob*100:.2f}%")
                elif final_prob >= 0.4:
                    label = "MODERATE chance of acceptance"
                    st.info(f"ℹ️ {label} — {final_prob*100:.2f}%")
                else:
                    label = "LOW chance of acceptance"
                    st.error(f"❌ {label} — {final_prob*100:.2f}%")
                
                # SHAP-style breakdown data for visualization and suggestions
                contributions = {
                    'Base Model (text)': text_prob,
                    'Publications Boost': 0.15 if has_pubs else 0.0,
                    'Successful Grants Boost': 0.1 * (1 - np.exp(-0.3 * successful_grants)) if successful_grants > 0 else 0.0,
                    'Abstract Quality Boost': 0.2 * abstract_quality,
                    'Funding Appropriateness Boost': 0.1 * funding_appropriateness
                }
                
                # For display convert to percents
                contributions_pct = {k: v*100 for k, v in contributions.items()}
                
                # show waterfall / bar chart of contributions
                st.write("---")
                st.subheader("Model Contribution Breakdown")
                labels = list(contributions_pct.keys())
                values = [contributions_pct[l] for l in labels]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{val:.1f}%" for val in values],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Base model score and metadata boosts (percent points)",
                    yaxis_title="Percent points",
                    xaxis_tickangle=-45,
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # If high acceptance probability, show project setup button
                if final_prob >= 0.65:
                    st.write("---")
                    st.subheader("Next Steps")
                    st.write("Your proposal has a high chance of acceptance! Set up your project details below:")
                    
                    # Store proposal details in session state
                    st.session_state.show_project_setup = True
                    st.session_state.proposal_details = {
                        'title': curr_title,
                        'abstract': abstract,
                        'researcher_name': researcher_name,
                        'institution': institution,
                        'funding_amount': funding_amount,
                        'predicted_acceptance': final_prob
                    }
                    
                    # Add a notice to open the Project Setup page (button to show)
                    if st.button("Set Up Project"):
                        st.rerun()
                
                # NEW: If low acceptance, show SHAP-style suggestions and auto-rewrite
                if final_prob < 0.40:
                    st.write("---")
                    st.subheader("🔍 Suggestions to Improve Your Proposal (SHAP-style Insights)")
                    
                    # Important keywords to check for
                    important_keywords = [
                        "innovative", "novel", "collaborative", "methodology", "impact",
                        "framework", "validated", "rigorous", "systematic", "transformative",
                        "approach", "strategy", "evaluation", "dissemination", "benchmarks"
                    ]
                    
                    missing_keywords = get_missing_keywords(combined_text, important_keywords)
                    
                    improvement_points = []
                    
                    # 1. Abstract quality issues
                    if abstract_quality < 0.4:
                        improvement_points.append("• **Improve abstract structure** — Add clearer methodology, step-by-step plan or phases, and explicit objectives and deliverables.")
                    
                    # 2. Missing high-impact keywords
                    if missing_keywords:
                        improvement_points.append(f"• **Add strong scientific keywords** that are currently missing: {', '.join(missing_keywords[:7])}. Use them naturally in objectives/methodology/outcomes.")
                    
                    # 3. Publications boost missing
                    if not has_pubs:
                        improvement_points.append("• **No publications noted** — if you have prior papers or conference abstracts, reference them. If not, emphasize team expertise and collaborations.")
                    
                    # 4. Funding mismatch
                    if funding_appropriateness < 0.6:
                        improvement_points.append("• **Funding requested may be misaligned** — justify budget line-items clearly or consider requesting an amount more typical for your project scope.")
                    
                    # 5. Previous grants very low
                    if successful_grants == 0:
                        improvement_points.append("• **No prior successful grants** — highlight institutional support, mentorship, or pilot data to build credibility.")
                    
                    # 6. Abstract length too short
                    if len(abstract.strip()) < 120:
                        improvement_points.append("• **Abstract is short / under-detailed** — expand motivation, methodology, expected outcomes and validation plan (aim for 150–300 words).")
                    
                    # 7. Check methodology phrases
                    method_patterns = [r"methodolog", r"approach", r"phase", r"experiment", r"evaluate", r"validation", r"benchmark"]
                    if not any(re.search(p, combined_text.lower()) for p in method_patterns):
                        improvement_points.append("• **Methodology details are lacking** — specify experiments, datasets, evaluation metrics and success criteria.")
                    
                    # Display suggestions ordered by severity (heuristic)
                    if improvement_points:
                        for p in improvement_points:
                            st.info(p)
                    else:
                        st.info("Your proposal needs more clarity and stronger justification. Try refining your abstract and methodology.")
                    
                    # Show a prioritized checklist (quick wins)
                    st.write("#### Quick Wins (apply these first):")
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
                    
                    # Provide suggested abstract (auto rewrite) and let user copy it
                    st.write("---")
                    st.subheader("✍️ Suggested Abstract (editable)")
                    suggested = generate_suggested_abstract(abstract, curr_title, missing_keywords, include_publication_note=has_pubs)
                    
                    # Show suggested abstract in an editable text area so user can copy/modify
                    user_edit = st.text_area("Suggested Abstract (edit as needed)", value=suggested, height=360, key="suggested_text_area_inside_form")
                    
                    # Save suggestion and context to session state so the outside button can use it
                    st.session_state['suggested_abstract'] = user_edit
                    st.session_state['last_title'] = curr_title
                    st.session_state['last_researcher'] = researcher_name
                    st.session_state['last_institution'] = institution
                    st.session_state['last_funding'] = funding_amount
                    st.session_state['last_prob'] = final_prob
                # end if final_prob < 0.40
    # end with st.form("prediction_form")
    
    # ---------- PLACE THE BUTTON OUTSIDE THE FORM (at bottom of Predict Acceptance page) ----------
    st.write("---")
    st.write("### Suggestion Utilities")
    if st.session_state.get('suggested_abstract'):
        st.markdown("A suggested abstract is available from your last prediction run.")
        if st.button("Use Suggested Abstract for New Prediction (Fill Predict Form)"):
            # Fill the Predict form defaults via session_state.proposal_details and rerun
            st.session_state.proposal_details = {
                'title': st.session_state.get('last_title', ''),
                'abstract': st.session_state.get('suggested_abstract', ''),
                'researcher_name': st.session_state.get('last_researcher', ''),
                'institution': st.session_state.get('last_institution', ''),
                'funding_amount': st.session_state.get('last_funding', 0.0),
                'predicted_acceptance': st.session_state.get('last_prob', 0)
            }
            st.success("Suggested abstract saved to the Predict form. The form will be reloaded with these values.")
            st.rerun()
    else:
        st.write("Run a prediction to generate tailored suggestions and a suggested abstract.")

elif page == "Project Setup":
    st.subheader("Project Setup")
    st.write("Set up a new project with budget allocation and phases.")
    
    projects = get_projects()
    
    if projects.empty:
        st.info("No projects found. Create a project in the Predict Acceptance page after getting a high acceptance probability.")
    else:
        st.write("### Existing Projects")
        for _, project in projects.iterrows():
            with st.expander(f"Project {project['id']}: {project['title']}"):
                col1, col2 = st.columns(2)
                col1.write(f"**Researcher:** {project['researcher_name']}")
                col1.write(f"**Institution:** {project['institution']}")
                col1.write(f"**Budget:** ${project['total_budget']:,.2f}")
                col1.write(f"**Status:** {project['status']}")
                col1.write(f"**Dates:** {project['start_date']} to {project['end_date']}")
                
                col2.write(f"**Acceptance Probability:** {project['predicted_acceptance']*100:.2f}%")
                
                # Show phases
                phases = get_project_phases(project['id'])
                if not phases.empty:
                    col2.write("**Project Phases:**")
                    for _, phase in phases.iterrows():
                        col2.write(f"- {phase['phase_name']}: ${phase['allocated_budget']:,.2f}")

elif page == "Project Tracking":
    st.subheader("Project Tracking")
    projects = get_projects()
    
    if projects.empty:
        st.info("No projects found. Create a project in the Predict Acceptance page.")
    else:
        project_id = st.selectbox("Select a project to track", projects['id'])
        
        if project_id:
            project = projects[projects['id'] == project_id].iloc[0]
            st.write(f"### Tracking: {project['title']}")
            
            # Project details
            col1, col2 = st.columns(2)
            col1.write(f"**Researcher:** {project['researcher_name']}")
            col1.write(f"**Institution:** {project['institution']}")
            col1.write(f"**Total Budget:** ${project['total_budget']:,.2f}")
            col1.write(f"**Status:** {project['status']}")
            col1.write(f"**Dates:** {project['start_date']} to {project['end_date']}")
            
            # Budget utilization
            budget_info = get_budget_utilization(project_id)
            col2.write(f"**Budget Used:** ${budget_info['total_used']:,.2f}")
            col2.write(f"**Budget Remaining:** ${budget_info['remaining']:,.2f}")
            col2.write(f"**Utilization:** {budget_info['utilization_percent']:.1f}%")
            
            # Budget gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=budget_info['utilization_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Budget Utilization %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 90], 'color': "gray"},
                           {'range': [90, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 95}}
            ))
            col2.plotly_chart(fig, use_container_width=True)
            
            # Detect and show issues
            issues = detect_project_issues(project_id)
            if issues:
                st.write("### Project Issues")
                for issue in issues:
                    if issue['severity'] == 'high':
                        st.error(f"**{issue['type'].title()}:** {issue['message']}")
                    elif issue['severity'] == 'medium':
                        st.warning(f"**{issue['type'].title()}:** {issue['message']}")
            
            # Progress tracking form
            st.write("### Add Progress Update")
            with st.form("progress_form"):
                # Get project phases
                phases = get_project_phases(project_id)
                if phases.empty:
                    st.info("No phases defined for this project.")
                else:
                    phase_options = {f"{phase['phase_name']} (${phase['allocated_budget']:,.2f})": phase['id'] for _, phase in phases.iterrows()}
                    selected_phase = st.selectbox("Select Phase", options=list(phase_options.keys()))
                    phase_id = phase_options[selected_phase]
                    
                    update_text = st.text_area("Update Description", height=150)
                    progress_percentage = st.slider("Progress Percentage", 0, 100, 0)
                    
                    # Get budget used so far for this phase
                    phase_updates = get_project_updates(project_id)
                    phase_budget_used = 0
                    if not phase_updates.empty:
                        phase_updates_for_phase = phase_updates[phase_updates['phase_id'] == phase_id]
                        if not phase_updates_for_phase.empty:
                            phase_budget_used = phase_updates_for_phase['budget_used'].sum()
                    
                    # Get allocated budget for this phase
                    phase_info = phases[phases['id'] == phase_id].iloc[0]
                    allocated_budget = phase_info['allocated_budget']
                    
                    budget_used = st.number_input(
                        "Budget Used in This Update", 
                        min_value=0.0, 
                        max_value=allocated_budget - phase_budget_used if allocated_budget - phase_budget_used > 0 else 0.0,
                        step=100.0,
                        value=0.0
                    )
                    
                    submitted = st.form_submit_button("Submit Update")
                    
                    if submitted:
                        if update_text.strip():
                            add_project_update(project_id, phase_id, update_text, progress_percentage, budget_used)
                            st.success("Update added successfully!")
                            st.rerun()
                        else:
                            st.warning("Please provide an update description.")
            
            # Display updates
            st.write("### Progress History")
            updates = get_project_updates(project_id)
            
            if updates.empty:
                st.info("No updates yet.")
            else:
                # Progress chart
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
                
                # Budget utilization chart
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
                
                # Update details
                for _, update in updates.iterrows():
                    with st.expander(f"Update on {update['update_date']} - {update['phase_name']}"):
                        st.write(f"**Progress:** {update['progress_percentage']}%")
                        st.write(f"**Budget Used:** ${update['budget_used']:,.2f}")
                        st.write(update['update_text'])

elif page == "Reports":
    st.subheader("Grant Management Reports")
    projects = get_projects()
    
    if projects.empty:
        st.info("No projects found. Create a project in the Predict Acceptance page.")
    else:
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Projects", len(projects))
        col2.metric("Active Projects", len(projects[projects['status'] == 'active']))
        col3.metric("Completed Projects", len(projects[projects['status'] == 'completed']))
        
        total_budget = projects['total_budget'].sum()
        total_used = sum(get_budget_utilization(pid)['total_used'] for pid in projects['id'])
        col4.metric("Total Budget Utilization", f"{(total_used/total_budget*100):.1f}%" if total_budget>0 else "0.0%")
        
        # Project status distribution
        st.write("### Project Status Distribution")
        status_counts = projects['status'].value_counts().reset_index()
        status_counts.columns = ['index', 'status']
        fig = px.pie(status_counts, values='status', names='index', title="Project Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Budget utilization by project
        st.write("### Budget Utilization by Project")
        budget_data = []
        for _, project in projects.iterrows():
            budget_info = get_budget_utilization(project['id'])
            budget_data.append({
                'Project': project['title'],
                'Budget Utilization %': budget_info['utilization_percent'],
                'Status': project['status']
            })
        
        budget_df = pd.DataFrame(budget_data)
        fig = px.bar(
            budget_df, 
            x='Project', 
            y='Budget Utilization %',
            color='Status',
            title="Budget Utilization by Project"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Issues report
        st.write("### Project Issues Report")
        all_issues = []
        for project_id in projects['id']:
            issues = detect_project_issues(project_id)
            for issue in issues:
                project_title = projects[projects['id'] == project_id].iloc[0]['title']
                all_issues.append({
                    'Project': project_title,
                    'Type': issue['type'],
                    'Severity': issue['severity'],
                    'Message': issue['message']
                })
        
        if all_issues:
            issues_df = pd.DataFrame(all_issues)
            st.dataframe(issues_df)
            
            # Issues by type
            st.write("#### Issues by Type")
            issues_by_type = issues_df.groupby('Type').size().reset_index(name='Count')
            fig = px.bar(issues_by_type, x='Type', y='Count', title="Issues by Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No issues detected across all projects!")
