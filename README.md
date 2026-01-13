# Cardiovascular Disease Prediction Web App

## Project Overview

This project is a machine-learning web application that predicts the presence of cardiovascular disease from clinical features. It includes data-preparation and model-training code, an evaluated ensemble/ML pipeline, a saved trained model, and a Flask web interface for making predictions.

## Problem Statement

Cardiovascular disease is a leading cause of mortality. Given patient features (age, sex, chest pain, blood pressure, cholesterol, etc.), the goal is to predict whether a patient has cardiovascular disease (binary classification) so clinicians or users can identify high-risk patients.

## Dataset

- File: `cardiovascular disease data.csv`
- `features.txt` lists feature descriptions

## Approach

- Data preprocessing and scaling.
- Training multiple models (Random Forest, Logistic Regression, Naive Bayes, KNN, XGBoost) and evaluating with cross-validation (see `main.py`).
- A Random Forest model is trained and used in the Flask app (`app.py`) to serve predictions.

## Outcomes

- `main.py` performs model evaluation and plots model accuracies.
- `app.py` runs a Flask app exposing a web form (templates/index.html) to collect feature values and show prediction results (templates/result.html).
- `trained_model.sav` is a persisted model artifact (if present).

## Files of interest

- [app.py](app.py) — Flask web application for serving predictions.
- [main.py](main.py) — Model training and evaluation script.
- [cardiovascular disease data.csv](cardiovascular disease data.csv) — Dataset.
- [trained_model.sav](trained_model.sav) — Saved model (ignored by default in `.gitignore`).
- [templates/index.html](templates/index.html) and [templates/result.html](result.html) — UI templates.
- [static/CSS/styles.css](static/CSS/styles.css) — Styling for the site.

## Run locally

1. Create and activate a virtual environment:

```bash
python -m venv env
# Windows PowerShell
.\env\Scripts\Activate.ps1
# or cmd
.\env\Scripts\activate.bat
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Open http://127.0.0.1:5000/ in your browser.

To retrain or re-evaluate models, run:

```bash
python main.py
```

## Deploying to GitHub / Heroku

- A `Procfile` is present for Heroku deployments. To deploy, create a GitHub repository and connect it to Heroku or push directly using Heroku Git.

## Git / GitHub Steps (recommended)

1. Initialize a git repo (if not already): `git init`.
2. Add and commit files: `git add . && git commit -m "Initial commit"`.
3. Create a repo on GitHub and add it as a remote: `git remote add origin <GIT_URL>`.
4. Push: `git push -u origin main` (or `master` depending on your branch).

If you prefer using GitHub CLI:

```bash
gh repo create <your-username>/<repo-name> --public --source=. --remote=origin --push
```

## Notes

- The app uses `app.py` as the Flask entrypoint. If you change the model or features, ensure `templates/index.html` form fields match feature order used by the model.
- `trained_model.sav` is intentionally in `.gitignore` to avoid committing large binary artifacts; instead consider using Git LFS or re-training on deploy.

## Contact

If you want, I can: initialize git, commit these files, and help create/push a GitHub repo (I will need the repository URL or permission to use the `gh` CLI).
