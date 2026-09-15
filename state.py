from typing import TypedDict


class MatchState(TypedDict):
    """Shared state passed between nodes in the candidate-match graph."""

    candidate_skills: list[str]
    candidate_experience_years: int
    job_requirements: list[str]
    match_percentage: int
    strengths: list[str]
    missing_skills: list[str]
    recommendation: str
