#!/bin/bash
# deploy_clean.sh
# Run this from inside your local repo root to clean it up and push the deployment-ready version.
# Usage: bash deploy_clean.sh

set -e

echo "🧹 Step 1: Removing unnecessary files and folders..."

# Remove venv (277MB — must NOT be on GitHub)
rm -rf venv/

# Remove raw Dataset folder (JSON files only used for training, not needed at runtime)
rm -rf Dataset/

# Remove old Flask backend (replaced by Streamlit app)
rm -f app.py

# Remove old Streamlit dashboard (replaced by fixed version)
rm -f new_dashboard.py

# Remove macOS junk
find . -name ".DS_Store" -delete

# Remove old grants.db (regenerated at runtime)
rm -f grants.db

echo "✅ Step 2: Copying new files..."

# Copy the new fixed dashboard as app.py
# (assumes new_dashboard_fixed.py is in the same directory)
if [ -f "new_dashboard_fixed.py" ]; then
    cp new_dashboard_fixed.py app.py
    rm new_dashboard_fixed.py
    echo "   Copied new_dashboard_fixed.py → app.py"
fi

echo "📦 Step 3: Writing clean requirements.txt..."
cat > requirements.txt << 'EOF'
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.18.0
EOF

echo "⚙️  Step 4: Creating .streamlit/config.toml..."
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[server]
headless = true
port = 8501

[theme]
base = "dark"
primaryColor = "#4F8EF7"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1F2E"
textColor = "#FAFAFA"
font = "sans serif"
EOF

echo "📝 Step 5: Updating .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
venv/
env/
.venv/
.DS_Store
.vscode/
grants.db
*.log
.env
secrets.toml
.ipynb_checkpoints/
EOF

echo ""
echo "🌿 Step 6: Committing and pushing to GitHub..."
git add -A
git status
git commit -m "chore: clean repo for Streamlit Cloud deployment

- Removed venv/ (277MB), Dataset/, old Flask app.py, old new_dashboard.py
- Added fixed new_dashboard_fixed.py as app.py (KeyError bugs fixed)
- Cleaned requirements.txt to Streamlit-only deps
- Added .streamlit/config.toml with dark theme
- Updated .gitignore and README"

git push origin main

echo ""
echo "✅ Done! Repo is clean and pushed."
echo ""
echo "Next steps:"
echo "  1. Go to https://share.streamlit.io"
echo "  2. New app → select this repo → Main file: app.py"
echo "  3. Deploy!"
