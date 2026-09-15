# Quinta AI Agent

FastAPI microservice hosting the LangGraph candidate-to-job matching workflow.

## Run locally

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Set `GOOGLE_API_KEY` in the process environment.
4. Start the API with `uvicorn main:app --host 0.0.0.0 --port 8000`.

The .NET backend calls `POST /api/ai/analyze-match`. Health checks can use
`GET /health`.
