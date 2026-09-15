import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from agent_graph import match_graph
from state import MatchState


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quinta AI Semantic Digital Twin",
    version="1.0.0",
    description="LangGraph-powered real-time candidate-to-job match analysis.",
)


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_skills: list[str] = Field(min_length=1)
    candidate_experience_years: int = Field(ge=0, le=80)
    job_requirements: list[str] = Field(min_length=1)


class MatchResponse(BaseModel):
    match_percentage: int = Field(ge=0, le=100)
    strengths: list[str]
    missing_skills: list[str]
    recommendation: str


@app.get("/health", tags=["Operations"])
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/api/ai/analyze-match",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    tags=["AI Match"],
)
async def analyze_match(request: MatchRequest) -> MatchResponse:
    initial_state: MatchState = {
        "candidate_skills": request.candidate_skills,
        "candidate_experience_years": request.candidate_experience_years,
        "job_requirements": request.job_requirements,
        "match_percentage": 0,
        "strengths": [],
        "missing_skills": [],
        "recommendation": "",
    }

    try:
        final_state = await match_graph.ainvoke(initial_state)
        return MatchResponse.model_validate(final_state)
    except Exception as exc:
        # Keep provider details in service logs without leaking them to API consumers.
        logger.exception("LangGraph candidate-match workflow failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI match-analysis workflow could not complete.",
        ) from exc
