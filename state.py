from typing import Any, NotRequired, TypedDict


class MatchState(TypedDict):
    """State shared by the evaluation, policy, and sanitization agents."""

    candidate_data: dict[str, Any]
    job_data: dict[str, Any]
    match_score: int
    analysis: str
    breakdown: NotRequired[dict[str, int]]
    strengths: NotRequired[list[str]]
    missing_skills: NotRequired[list[str]]
    ai_recommendation: NotRequired[str]
    policy_flag: bool
    policy_feedback: str
