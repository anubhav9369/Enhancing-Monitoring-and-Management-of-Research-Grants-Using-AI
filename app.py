from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import sqlite3


# FLASK APP SETUP
app = Flask(__name__)
CORS(app)

# EMAIL CONFIGURATION  (REPLACE PASSWORD)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'anubhav.verma2024@vitstudent.ac.in'
app.config['MAIL_PASSWORD'] = 'vrwdomqoggjzmdhm'   # <- paste NEW APP PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = ('Grant AI System', 'anubhav.verma2024@vitstudent.ac.in')

mail = Mail(app)

# DATABASE SETUP
def init_db():
    conn = sqlite3.connect('grants.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            researcher TEXT,
            total_budget REAL,
            start_date TEXT,
            end_date TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            progress INTEGER,
            budget_used REAL,
            summary TEXT,
            update_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def query_db(query, args=(), one=False):
    conn = sqlite3.connect('grants.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    conn.commit()
    conn.close()
    return (rv[0] if rv else None) if one else rv


# SEND EMAIL ALERT
def send_alert_email(project, alerts):
    """Send an email alert."""
    if not alerts:
        return

    msg_body = f"""
🚨 Project Alert!

Title: {project['title']}
Researcher: {project['researcher']}
End Date: {project['end_date']}
Total Budget: ₹{project['total_budget']:,.2f}

Detected Issues:
- {chr(10).join(alerts)}

Please review the project immediately.
"""

    try:
        msg = Message(
            subject=f"[ALERT] Project '{project['title']}' requires attention",
            recipients=["anubhav.verma2024@vitstudent.ac.in"],
            body=msg_body
        )
        mail.send(msg)
        print(f"📧 Email sent for project: {project['title']}")

    except Exception as e:
        print("❌ Email sending failed:", e)


# API ENDPOINTS

@app.route('/projects', methods=['POST'])
def create_project():
    try:
        data = request.get_json(force=True)
        query_db('''
            INSERT INTO projects (title, researcher, total_budget, start_date, end_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['title'], data['researcher'], float(data['budget']), data['start'], data['end']))
        return jsonify({'message': 'Project created successfully'}), 201
    except Exception as e:
        print("❌ Error in create_project:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/projects', methods=['GET'])
def get_projects():
    projects = query_db('SELECT * FROM projects ORDER BY id DESC')
    return jsonify([dict(p) for p in projects])


@app.route('/projects/<int:pid>/update', methods=['POST'])
def add_update(pid):
    try:
        data = request.get_json(force=True)
        query_db('''
            INSERT INTO updates (project_id, progress, budget_used, summary)
            VALUES (?, ?, ?, ?)
        ''', (pid, int(data['progress']), float(data['budget_used']), data['summary']))
        return jsonify({'message': 'Update added successfully'}), 201
    except Exception as e:
        print("❌ Error in add_update:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/projects/<int:pid>/status', methods=['GET'])
def project_status(pid):
    """Return project tracking data and send alerts if needed."""
    project = query_db('SELECT * FROM projects WHERE id=?', (pid,), one=True)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    updates = query_db('SELECT * FROM updates WHERE project_id=? ORDER BY update_date ASC', (pid,))
    total_spent = sum([u['budget_used'] for u in updates]) if updates else 0
    avg_progress = sum([u['progress'] for u in updates]) / len(updates) if updates else 0

    alerts = []
    today = datetime.now().date()
    end_date = datetime.strptime(project['end_date'], "%Y-%m-%d").date()

    # Checks:
    if total_spent > project['total_budget']:
        alerts.append("⚠️ Budget overspent.")

    if avg_progress < 50 and total_spent > (0.7 * project['total_budget']):
        alerts.append("⏰ Low progress despite high fund usage.")

    if today > end_date and avg_progress < 100:
        alerts.append("🚨 Project is overdue and incomplete.")

    if updates:
        last_update_date = datetime.strptime(updates[-1]['update_date'], "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - last_update_date).days > 30:
            alerts.append(f"🕒 No updates for {(datetime.now() - last_update_date).days} days.")
    else:
        alerts.append("⚠️ No progress updates submitted yet.")

    # FIX: Application context required
    if alerts:
        with app.app_context():
            send_alert_email(project, alerts)

    remaining_budget = project['total_budget'] - total_spent

    return jsonify({
        'project': dict(project),
        'total_spent': total_spent,
        'avg_progress': avg_progress,
        'remaining_budget': remaining_budget,
        'budget_utilization': (total_spent / project['total_budget'] * 100),
        'alerts': alerts,
        'updates': [dict(u) for u in updates]
    })


# BACKGROUND SCHEDULER (fix: app context)
def check_overdue_projects():
    with app.app_context():   # REQUIRED FIX
        print("🔍 Running scheduled project health check...")

        conn = sqlite3.connect('grants.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM projects WHERE status='active'")
        projects = c.fetchall()

        for project in projects:
            pid = project['id']
            end_date = datetime.strptime(project['end_date'], "%Y-%m-%d").date()
            today = datetime.now().date()

            c.execute("SELECT * FROM updates WHERE project_id=?", (pid,))
            updates = c.fetchall()

            total_spent = sum([u['budget_used'] for u in updates]) if updates else 0
            avg_progress = sum([u['progress'] for u in updates]) / len(updates) if updates else 0

            alerts = []

            if total_spent > project['total_budget']:
                alerts.append("⚠️ Budget overspent.")

            if avg_progress < 50 and total_spent > (0.7 * project['total_budget']):
                alerts.append("⏰ Low progress despite high fund usage.")

            if today > end_date and avg_progress < 100:
                alerts.append("🚨 Project is overdue and incomplete.")

            if not updates:
                alerts.append("⚠️ No progress updates submitted yet.")
            else:
                last_update = datetime.strptime(updates[-1]['update_date'], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last_update).days > 30:
                    alerts.append(f"🕒 No updates for {(datetime.now() - last_update).days} days.")

            if alerts:
                send_alert_email(project, alerts)

        conn.close()


# Scheduler: runs every 1 minute
scheduler = BackgroundScheduler()
scheduler.add_job(check_overdue_projects, 'interval', minutes=1)
scheduler.start()


# MANUAL TRIGGER
@app.route('/test-alert')
def test_alert():
    check_overdue_projects()
    return "Manual alert check executed"


# RUN
if __name__ == '__main__':
    app.run(debug=True)
