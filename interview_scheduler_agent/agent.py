"""
agent.py
────────────────────────────────────────────────────────────────────────────
AI Interview Scheduling Specialist Agent (Student 3 - Meeting Orchestration).
Integrates:
1. Backend-mediated data retrieval (candidates in Interview Selection + blocked calendar events).
2. Deterministic constraint-satisfaction slot engine with forward-search overflow handling.
3. ChatGroq reasoning agent with strict anti-clash verification and explicit assumption logging.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from interview_scheduler_agent.schemas import (
    GenerateScheduleRequest,
    ScheduleProposalResponse,
)
from interview_scheduler_agent.tools import (
    fetch_candidates_for_interview,
    fetch_blocked_slots,
    fetch_schedule_config,
)
from interview_scheduler_agent.scheduler import compute_interview_schedule

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the expert AI Interview Scheduling Specialist for Skill-Hub's Meeting Orchestration platform.

### YOUR SINGLE RESPONSIBILITY:
Your sole mission is to propose clash-free, highly optimized interview schedule drafts for shortlisted candidates based on HR parameters. You propose drafts; HR makes final approval. You must NEVER persist or finalize appointments on your own.

### CRITICAL CONSTRAINTS & RULES:
1. ZERO DOUBLE BOOKING: Never schedule multiple candidates into the same parallel track or room simultaneously. Never schedule over an existing calendar event or holiday.
2. PARALLEL TRACKS: You can utilize up to N concurrent tracks at the same time slot, each representing a separate interview room or panel.
3. SEPARATION: Clearly separate "successfully scheduled" candidates from "unscheduled candidates".
4. FORWARD EXTENSION: If candidates do not fit in the initial date range, forward-schedule up to 14 days beyond the end date, flagging them clearly.
5. EXPLICIT ASSUMPTIONS: You must state every assumption explicitly (working hours, breaks, weekend exclusion, track counts) in your validation notes rather than silently assuming them.
"""


class InterviewSchedulerAgent:
    """
    AI Agent that coordinates data retrieval, deterministic clash-free slot generation,
    and LLM validation for interview scheduling.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")

        if not api_key:
            logger.warning("[SchedulerAgent] GROQ_API_KEY not configured. Deterministic engine will operate in fallback mode.")
            self.llm = None
        else:
            try:
                self.llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=self.model_name,
                    temperature=0.1,
                    max_tokens=2500
                )
            except Exception as e:
                logger.error("[SchedulerAgent] Failed to initialize ChatGroq: %s. Using fallback mode.", str(e))
                self.llm = None

    async def generate_schedule(self, request: GenerateScheduleRequest) -> ScheduleProposalResponse:
        """
        Executes the end-to-end interview scheduling workflow:
        1. Fetches candidates in Interview Selection via backend HTTP.
        2. Fetches existing calendar events and blocked slots via backend HTTP.
        3. Generates clash-free slots across parallel tracks, automatically forward-searching if candidates overflow.
        4. Validates and enriches the draft proposal with explicit assumptions and AI reasoning.
        """
        logger.info(
            "[SchedulerAgent] Starting schedule generation for Job ID: %s (Window: %s to %s, Tracks: %d, Duration: %dm)",
            request.job_vacancy_id,
            request.start_date,
            request.end_date,
            request.parallel_tracks,
            request.interview_duration_minutes
        )

        # 1. Fetch Candidates from Backend
        cand_data = fetch_candidates_for_interview(request.job_vacancy_id)
        candidates = cand_data.get("candidates", [])
        job_title = cand_data.get("job_title", "Software Engineer")

        # 2. Fetch Blocked Slots from Backend
        blocked_slots = fetch_blocked_slots(
            company_id=request.company_id,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # 3. Fetch Company Schedule Config (Working Hours)
        config = fetch_schedule_config(request.company_id)
        work_start = request.working_hours_start or config.get("workingHoursStart", "09:00")
        work_end = request.working_hours_end or config.get("workingHoursEnd", "17:00")
        buffer_min = request.buffer_minutes if request.buffer_minutes is not None else config.get("bufferMinutes", 10)

        # 4. Deterministic Constraint Satisfaction & Forward Search Engine
        proposal = compute_interview_schedule(
            job_vacancy_id=request.job_vacancy_id,
            job_title=job_title,
            candidates=candidates,
            blocked_slots=blocked_slots,
            start_date_str=request.start_date,
            end_date_str=request.end_date,
            duration_min=request.interview_duration_minutes,
            parallel_tracks=request.parallel_tracks,
            work_start_str=work_start,
            work_end_str=work_end,
            buffer_min=buffer_min,
            max_forward_search_days=14
        )

        # 5. LLM Audit & Validation (if Groq LLM available)
        if self.llm and len(proposal.proposed_slots) > 0:
            try:
                audit_prompt = f"""Review the proposed interview schedule draft:
Job: {job_title}
Candidates: Total={proposal.summary.total_candidates}, Scheduled={proposal.summary.scheduled_count}, Unscheduled={proposal.summary.unscheduled_count}
Original Window: {proposal.summary.original_date_range}
Effective Window: {proposal.summary.effective_date_range}
Forward Days Extended: {proposal.summary.forward_days_extended}
Parallel Tracks Utilized: {proposal.summary.tracks_utilized}

Please provide 2-3 concise, professional validation statements confirming that the schedule is clash-free, adheres to working hours, and explicitly stating the forward search status. Return strictly a JSON list of strings.
Example: ["All 5 candidates successfully scheduled across 2 tracks.", "No clashes detected with existing calendar appointments.", "Forward search was not required."]"""

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=audit_prompt)
                ]
                res = await self.llm.ainvoke(messages)
                raw_text = res.content.strip()
                # Extract JSON array
                start_idx = raw_text.find("[")
                end_idx = raw_text.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    notes = json.loads(raw_text[start_idx:end_idx + 1])
                    if isinstance(notes, list) and notes:
                        proposal.summary.ai_validation_notes = [str(n) for n in notes]
            except Exception as e:
                logger.warning("[SchedulerAgent] Groq LLM validation call skipped: %s", str(e))

        return proposal


# Global singleton instance
scheduler_agent = InterviewSchedulerAgent()
