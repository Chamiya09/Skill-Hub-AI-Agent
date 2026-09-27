"""
schemas.py
────────────────────────────────────────────────────────────────────────────
Pydantic schemas for the AI Interview Slot Generator Agent.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateItem(BaseModel):
    candidate_id: str
    application_id: str
    full_name: str
    email: str
    score: Optional[float] = None


class BlockedSlotItem(BaseModel):
    id: str
    title: str
    date: str  # YYYY-MM-DD
    start_time: str  # HH:mm
    end_time: str  # HH:mm
    type: str = "CompanyEvent"  # CompanyEvent, Holiday, etc.


class ScheduleConfigRequest(BaseModel):
    working_hours_start: str = "09:00"
    working_hours_end: str = "17:00"
    buffer_minutes: int = 10
    timezone: str = "Asia/Colombo"


class GenerateScheduleRequest(BaseModel):
    job_vacancy_id: str = Field(..., description="UUID of the job vacancy")
    company_id: Optional[str] = Field(None, description="UUID of the hiring company")
    start_date: str = Field(..., description="Desired start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Desired end date (YYYY-MM-DD)")
    interview_duration_minutes: int = Field(30, description="Duration per interview session in minutes")
    parallel_tracks: int = Field(2, description="Number of concurrent interviewer panels / rooms")
    working_hours_start: Optional[str] = Field("09:00", description="Working day start time (HH:mm)")
    working_hours_end: Optional[str] = Field("17:00", description="Working day end time (HH:mm)")
    buffer_minutes: int = Field(10, description="Rest / transition buffer between consecutive slots")


class ProposedSlot(BaseModel):
    slot_id: str
    candidate_id: str
    candidate_name: str
    candidate_email: str
    date: str  # YYYY-MM-DD
    start_time: str  # HH:mm
    end_time: str  # HH:mm
    track_number: int  # 1, 2...
    track_name: str  # e.g. "Interview Track 1 (Room A)"
    is_extended_search: bool = False


class UnscheduledCandidate(BaseModel):
    candidate_id: str
    candidate_name: str
    candidate_email: str
    reason: str


class ScheduleSummary(BaseModel):
    total_candidates: int
    scheduled_count: int
    unscheduled_count: int
    original_date_range: str
    effective_date_range: str
    forward_days_extended: int
    tracks_utilized: int
    assumptions_made: List[str]
    ai_validation_notes: List[str]


class ScheduleProposalResponse(BaseModel):
    job_vacancy_id: str
    job_title: str
    proposed_slots: List[ProposedSlot]
    unscheduled_candidates: List[UnscheduledCandidate]
    summary: ScheduleSummary
    is_draft: bool = True
