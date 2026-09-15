import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from agent_graph import match_graph
from state import MatchState


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Skill Hu AI Semantic Digital Twin",
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
    policy_flag: bool
    policy_feedback: str


@app.get("/health", tags=["Operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/ai/analyze-match",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    tags=["AI Match"],
)
async def analyze_match(request: MatchRequest) -> MatchResponse:
    initial_state: MatchState = {
        "candidate_data": {
            "skills": request.candidate_skills,
            "experience_years": request.candidate_experience_years,
        },
        "job_data": {"requirements": request.job_requirements},
        "match_score": 0,
        "analysis": "",
        "policy_flag": False,
        "policy_feedback": "",
    }

    try:
        final_state = await match_graph.ainvoke(initial_state)
        # Preserve the established .NET wire contract while the graph keeps its
        # richer internal ethics and policy state.
        return MatchResponse(
            match_percentage=final_state["match_score"],
            strengths=[],
            missing_skills=[],
            recommendation=final_state["analysis"],
            policy_flag=final_state["policy_flag"],
            policy_feedback=final_state["policy_feedback"],
        )
    except Exception as exc:
        # Keep provider details in service logs without leaking them to API consumers.
        logger.exception("LangGraph candidate-match workflow failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI match-analysis workflow could not complete.",
        ) from exc


class BatchJobItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    title: str
    company: str
    location: str
    requirements: list[str] = Field(default_factory=list)


class BatchRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_skills: list[str] = Field(default_factory=list)
    candidate_digital_cv: dict[str, Any] = Field(default_factory=dict)
    jobs: list[BatchJobItem] = Field(default_factory=list)


class BatchJobRecommendationResult(BaseModel):
    job_id: str
    match_percentage: int = Field(ge=0, le=100)
    is_recommended: bool


@app.post(
    "/api/ai/batch-recommend",
    response_model=list[BatchJobRecommendationResult],
    status_code=status.HTTP_200_OK,
    tags=["AI Match"],
)
async def batch_recommend_jobs(request: BatchRecommendationRequest) -> list[BatchJobRecommendationResult]:
    """Score candidate against a batch of open jobs using semantic matching."""
    if not request.jobs:
        return []

    results: list[BatchJobRecommendationResult] = []

    # Candidate data incorporates both explicit skills list and any extended digital CV data
    candidate_data = dict(request.candidate_digital_cv)
    if "skills" not in candidate_data:
        candidate_data["skills"] = request.candidate_skills

    for job in request.jobs:
        job_data = {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "requirements": job.requirements,
        }

        initial_state: MatchState = {
            "candidate_data": candidate_data,
            "job_data": job_data,
            "match_score": 0,
            "analysis": "",
            "policy_flag": False,
            "policy_feedback": "",
        }

        try:
            final_state = await match_graph.ainvoke(initial_state)
            score = max(0, min(100, int(final_state.get("match_score", 0))))
        except Exception as exc:
            logger.warning("Error evaluating job %s in batch recommendation: %s", job.job_id, exc)
            score = 0

        results.append(
            BatchJobRecommendationResult(
                job_id=job.job_id,
                match_percentage=score,
                is_recommended=(score >= 80),
            )
        )

    return results

