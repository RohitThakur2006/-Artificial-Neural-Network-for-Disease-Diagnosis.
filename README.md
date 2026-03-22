# Artificial Neural Network for Disease Diagnosis

## Deploy on Railway

This project is ready to deploy on Railway with:
- `requirements.txt` (Python dependencies)
- `Procfile` (start command with Gunicorn)
- `runtime.txt` (Python version)

### Steps

1. Push your project to a GitHub repository.
2. Go to [https://railway.app](https://railway.app) and sign in.
3. Click **New Project** → **Deploy from GitHub repo**.
4. Select this repository and allow Railway to import it.
5. Railway will auto-detect Python and install from `requirements.txt`.
6. In your service settings, ensure these values are used:
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers ${WORKERS:-2}`
   - **Python Version**: `3.11` (already set by `runtime.txt`)
7. Deploy the service.
8. Open the generated Railway domain URL to use the app.

### Optional environment variables

- `WORKERS`: Number of Gunicorn workers (default is `2` from `Procfile`).

### Troubleshooting

- If deployment fails during build, confirm `requirements.txt` is present and committed.
- If the app does not start, verify the entry point is `app:app` (Flask app object in `app.py`).
