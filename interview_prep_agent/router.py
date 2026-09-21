"""
router.py
────────────────────────────────────────────────────────────────────────────
FastAPI APIRouter for Student 1 Interview Preparation Guide.
Strictly handles routing and requests for:
  POST /api/student1/generate-guide
"""

import logging
from fastapi import APIRouter, HTTPException

from interview_prep_agent.interview_graph import generate_interview_guide
from interview_prep_agent.schemas import GenerateGuideRequest, InterviewGuideResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Interview Prep Agent"])


@router.post("/api/student1/generate-guide", response_model=InterviewGuideResponse)
@router.post("/api/interview-prep/generate-guide", response_model=InterviewGuideResponse)
async def generate_guide_endpoint(request: GenerateGuideRequest):
    """
    Analyzes the candidate's target job description using a sequential 3-agent
    workflow (Extractor -> Generator -> Validator) with Groq API (temperature=0.0).
    Returns strictly validated theoretical and practical study guidelines.
    """
    if not request.job_description or not request.job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description is required to generate interview preparation guidelines.",
        )

    title = request.target_role or request.job_title or "Software Engineer"
    company = request.company_name or "Enterprise Partner"
    level = request.experience_level or "Mid-Senior"

    try:
        logger.info(
            "[Student1 Router] Generating interview guide for '%s' at '%s'...",
            title,
            company,
        )
        response = await generate_interview_guide(
            job_title=title,
            job_description=request.job_description,
            experience_level=level,
            company_name=company,
        )
        return response
    except Exception as e:
        logger.error(
            "[Student1 Router] Failed to generate interview prep guide: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"AI Agent failed to generate study guidelines: {str(e)}",
        )
