# House Price Predictor API

Simple end-to-end MLOps demo project:

- **Model**: scikit-learn Linear Regression (house size -> price)
- **Tracking**: MLflow (params, metrics, model artifacts)
- **Serving**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions runs tests -> tested code promoted to `production` branch -> Back4app Containers auto-builds & deploys from that branch

## Try it

Once deployed, open `/docs` on the live Back4app URL for interactive Swagger UI.

Example request:

```bash
curl -X POST "<BACK4APP_URL>/predict" \
  -H "Content-Type: application/json" \
  -d '{"size_sqft": 1500}'
```

## Project structure

```
.
├── train.py                    # trains model, logs to MLflow, saves model/model.pkl
├── app.py                      # FastAPI app serving predictions
├── test_app.py                 # pytest test cases (run in CI)
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci-cd.yml # CI (tests) + CD (promote to production branch)
```

## Run locally

```bash
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

## Deployment flow

```
GitHub push (main branch)
        |
        v
GitHub Actions - test job
   - train.py runs
   - pytest runs
        |
        v (if passed)
GitHub Actions - promote-to-production job
   - force-pushes tested code to "production" branch
        |
        v
Back4app Containers (watching "production" branch)
   - detects new commit
   - builds Docker image from Dockerfile
   - deploys automatically, zero downtime
```

## One-time Back4app setup

1. Sign up at back4app.com (no credit card needed)
2. Dashboard -> Containers -> Create New App
3. Install the Back4app GitHub App and select this repository
4. Branch to deploy from: `production` (not `main`)
5. Root directory: `.` (Back4app auto-detects the Dockerfile)
6. Select the Free plan and deploy
