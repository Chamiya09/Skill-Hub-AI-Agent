from typing import Any, TypedDict


class MatchState(TypedDict):
    """State shared by the evaluation, policy, and sanitization agents."""

    candidate_data: dict[str, Any]
    job_data: dict[str, Any]
    match_score: int
    analysis: str
    policy_flag: bool
    policy_feedback: str
