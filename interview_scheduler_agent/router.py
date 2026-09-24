"""
router.py
────────────────────────────────────────────────────────────────────────────
FastAPI route for the AI Interview Slot Generator Agent (Student 3).
"""

from fastapi import APIRouter, HTTPException
from interview_scheduler_agent.schemas import (
    GenerateScheduleRequest,
    ScheduleProposalResponse,
)
from interview_scheduler_agent.agent import scheduler_agent
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Interview Scheduler Agent"])


@router.post(
    "/api/interview-scheduler/generate-schedule",
    response_model=ScheduleProposalResponse,
    summary="Generate clash-free interview schedule proposal",
    description="Proposes an optimal, non-overlapping multi-track schedule with forward search overflow extension."
)
async def generate_interview_schedule(request: GenerateScheduleRequest):
    try:
        proposal = await scheduler_agent.generate_schedule(request)
        return proposal
    except Exception as e:
        logger.error("[Router] Error in generate_interview_schedule: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the interview schedule: {str(e)}"
        )
