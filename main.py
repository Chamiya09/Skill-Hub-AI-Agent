from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import domain agent routers
from interview_prep_agent.router import router as interview_prep_router
from cv_evaluator_agent.router import router as cv_evaluator_router
from skill_assessment_agent.router import router as assessment_agent_router
from interview_scheduler_agent.router import router as interview_scheduler_router

# Explicitly load .env file from the current directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="AI Agent Services",
    description="Enterprise Multi-Agent microservices platform."
)

# Add CORS middleware so the ASP.NET Core backend can call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register domain agent routers
app.include_router(interview_prep_router)
app.include_router(cv_evaluator_router)
app.include_router(assessment_agent_router)
app.include_router(interview_scheduler_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Agent Backend!"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Agent Services"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
