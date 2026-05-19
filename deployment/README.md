# Phase 6: Model Deployment

## Deployment Strategy

**Online REST API** using FastAPI — serves real-time churn predictions over HTTP.

---

## Project Structure

```
├── app.py                  # FastAPI inference API
├── Dockerfile              # Container definition (Python 3.11)
├── requirements.txt        # Pinned Python dependencies
├── model/                  # MLflow model artifacts
│   ├── MLmodel
│   ├── model.pkl
│   ├── conda.yaml
│   └── python_env.yaml
├── tests/
│   └── test_api.py         # Pytest test suite
└── .github/
    └── workflows/
        └── ci_cd.yml       # GitHub Actions CI/CD pipeline
```

---

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | API info and expected feature list |
| GET    | `/health`  | Liveness probe                     |
| GET    | `/ready`   | Readiness probe (model loaded)     |
| POST   | `/predict` | Predict churn for one customer     |

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure": 12,
      "MonthlyCharges": 65.5,
      "TotalCharges": 786.0
    }
  }'
```

## Example Response

```json
{
  "prediction": 1,
  "churn_status": "Churn",
  "churn_probability": 0.7341,
  "risk_level": "High",
  "missing_features_filled": ["Contract", "PaymentMethod", "..."]
}
```

**Risk levels:**
- `High` — probability ≥ 0.70
- `Medium` — probability 0.40–0.69
- `Low` — probability < 0.40

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Open Swagger UI
http://localhost:8000/docs
```

---

## Run with Docker

```bash
# Build image
docker build -t telecom-churn-api .

# Run container
docker run -d --name churn-api -p 8000:8000 telecom-churn-api

# Check health
curl http://localhost:8000/health

# Stop container
docker stop churn-api
```

---

## Run Tests

```bash
pytest tests/test_api.py -v
```

---

## CI/CD Pipeline (GitHub Actions)

On every push to `main`, the pipeline:

1. **Test** — runs `pytest` against the API
2. **Build** — builds and pushes Docker image to Docker Hub
3. **Deploy** — SSHs into the server, pulls latest image, restarts container

### Required GitHub Secrets

| Secret             | Description                      |
|--------------------|----------------------------------|
| `DOCKER_USERNAME`  | Docker Hub username              |
| `DOCKER_PASSWORD`  | Docker Hub password or token     |
| `SERVER_HOST`      | Deployment server IP/hostname    |
| `SERVER_USER`      | SSH username on the server       |
| `SERVER_SSH_KEY`   | Private SSH key for the server   |

---

## Completed Checklist

- [x] FastAPI inference API (`/predict`, `/health`, `/ready`)
- [x] Churn probability + risk level in response
- [x] Input validation with missing/unknown column logging
- [x] Request and error logging
- [x] Docker containerization (Python 3.11, pinned deps)
- [x] Docker HEALTHCHECK configured
- [x] MLmodel artifact path fixed (relative, not Windows absolute)
- [x] Pytest test suite (7 tests)
- [x] GitHub Actions CI/CD pipeline
- [x] Swagger UI available at `/docs`
