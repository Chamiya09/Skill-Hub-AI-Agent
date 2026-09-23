"""
router.py
────────────────────────────────────────────────────────────────────────────
FastAPI router for the Skill Assessment Agent.
Endpoints:
  POST /api/assessment-agent/generate-question
"""

import logging
from fastapi import APIRouter, HTTPException

from skill_assessment_agent.agent import AssessmentAgent
from skill_assessment_agent.schemas import (
    GenerateQuestionRequest,
    GenerateQuestionResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Skill Assessment Agent"])
_agent_instance = None


def get_agent() -> AssessmentAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AssessmentAgent()
    return _agent_instance


@router.post(
    "/api/assessment-agent/generate-question",
    response_model=GenerateQuestionResponse,
    summary="Generate Calibrated Coding Challenge for a Job Requisition"
)
async def generate_question_endpoint(request: GenerateQuestionRequest):
    """
    Executes the single-agent technical assessment question generation workflow:
    1. Uses `fetch_job_vacancy_context` database tool to retrieve job requirements.
    2. Determines the best matching programming language from supported languages.
    3. Generates 1 calibrated moderate coding challenge with complete test cases,
       starter stub, and solution adhering to Judge0 sandbox limitations.
    """
    if not request.job_vacancy_id or not request.job_vacancy_id.strip():
        raise HTTPException(
            status_code=400,
            detail="job_vacancy_id is required."
        )

    try:
        agent = get_agent()
        response = await agent.generate_question(
            job_vacancy_id=request.job_vacancy_id,
            focus_area=request.focus_area,
            job_context=request.job_context.model_dump() if request.job_context else None
        )
        return response
    except Exception as e:
        logger.error("[Assessment Router] Failed to generate assessment question: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"AI Agent failed to generate assessment challenge: {str(e)}"
        )