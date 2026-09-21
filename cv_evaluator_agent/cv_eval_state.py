"""
cv_eval_state.py
────────────────────────────────────────────────────────────────────────────
LangGraph shared state for the CV Evaluation 3-Agent pipeline.

The state dict is passed sequentially through three nodes:
  Node 1 – extractor   : parses raw CV text into structured fields
  Node 2 – evaluator   : LLM scoring against the job description
  Node 3 – validator   : deterministic JSON validation + business rules

Only keys that a node MODIFIES are returned from that node;
LangGraph merges them back into the shared state automatically.
"""

from typing import TypedDict


class CvEvalState(TypedDict):
    """Shared pipeline state for the CV evaluation workflow."""

    # ── Inputs (populated before the graph runs) ──────────────────────────
    cv_text: str
    job_description: str

    # ── Agent 1 outputs ───────────────────────────────────────────────────
    candidate_name: str
    frontend_skills: list[str]
    backend_skills: list[str]
    years_of_experience: float
    frameworks_used: list[str]
    certifications: list[str]
    candidate_experience_summary: str
    project_complexities: str
    candidate_education_summary: str

    # ── Agent 2 outputs ───────────────────────────────────────────────────
    match_score: int
    strengths: list[str]
    missing_skills: list[str]
    recommendation: str

    # ── Agent 3 outputs ───────────────────────────────────────────────────
    validation_notes: list[str]
    final_match_score: int
