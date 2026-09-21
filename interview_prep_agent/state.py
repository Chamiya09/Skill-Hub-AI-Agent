"""
state.py
────────────────────────────────────────────────────────────────────────────
LangGraph shared state for the Student 1 sequential 3-agent pipeline.
"""

from typing import Any, Dict, List, TypedDict


class InterviewGuideState(TypedDict, total=False):
    """Shared pipeline state for the interview preparation guide workflow."""

    # ── Inputs ────────────────────────────────────────────────────────────
    job_title: str
    job_description: str
    experience_level: str
    company_name: str

    # ── Agent 1: The JD Analyzer (Extractor) ──────────────────────────────
    extracted_languages: List[str]
    extracted_frameworks: List[str]
    extracted_cloud_tools: List[str]
    extracted_domain_concepts: List[str]
    role_technical_summary: str

    # ── Agent 2: The Guide Architect (Generator) ──────────────────────────
    role_overview_summary: str
    theoretical_main_concepts: List[Dict[str, Any]]
    practical_implementation_guidelines: List[Dict[str, Any]]
    pro_tips: List[str]
    preparation_checklist: List[str]

    # ── Agent 3: The Strict Validator (QA) ────────────────────────────────
    key_theoretical_areas: List[Dict[str, Any]]
    practical_implementation_focus: List[Dict[str, Any]]
    validation_notes: List[str]
    is_valid: bool


# Alias for backwards compatibility
PrepState = InterviewGuideState
