"""
interview_graph.py
────────────────────────────────────────────────────────────────────────────
Sequential Multi-Agent workflow for the Student 1 Interview Prep Guide.
Chains:
  Agent 1: The JD Analyzer (Extractor)
  Agent 2: The Guide Architect (Generator)
  Agent 3: The Strict Validator (QA guardrail)
"""

import logging
import os
import re
import uuid
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from interview_prep_agent.prompts import (
    GUIDE_ARCHITECT_SYSTEM_PROMPT,
    JD_ANALYZER_SYSTEM_PROMPT,
)
from interview_prep_agent.schemas import (
    GuideGenerationResult,
    InterviewGuideResponse,
    JobDescriptionExtraction,
    StudyFocusArea,
)
from interview_prep_agent.state import InterviewGuideState

logger = logging.getLogger(__name__)


# ─── LLM Factory ───────────────────────────────────────────────────────────


def _get_structured_llm(schema) -> Any:
    """
    Creates a Groq Runnable with temperature=0.0 and structured output schema.
    Applies automatic fallback if the primary model encounters rate limits.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not configured.")

    primary_model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    fallback_model_name = os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")

    primary_llm = ChatGroq(
        model=primary_model_name,
        api_key=api_key,
        temperature=0.0,
        max_retries=1,
        timeout=75,
    ).with_structured_output(schema=schema, method="json_mode")

    fallback_llm = ChatGroq(
        model=fallback_model_name,
        api_key=api_key,
        temperature=0.0,
        max_retries=2,
        timeout=75,
    ).with_structured_output(schema=schema, method="json_mode")

    logger.info(
        "[InterviewGraph] LLM Chain: Primary=%s | Fallback=%s (temp=0.0)",
        primary_model_name,
        fallback_model_name,
    )

    return primary_llm.with_fallbacks([fallback_llm])


# ─── Agent 1: The JD Analyzer (Extractor) ───────────────────────────────────


async def extractor_node(state: InterviewGuideState) -> Dict[str, Any]:
    """
    Agent 1 – The JD Analyzer (Extractor).
    Analyzes the raw Job Description and extracts structured technical requirements.
    """
    job_desc = state.get("job_description", "")
    job_title = state.get("job_title", "Software Engineer")
    logger.info(
        "[Agent1/Extractor] Analyzing JD for '%s' (%d chars)...",
        job_title,
        len(job_desc),
    )

    structured_llm = _get_structured_llm(JobDescriptionExtraction.model_json_schema())

    user_prompt = (
        f"JOB TITLE: {job_title}\n"
        f"COMPANY: {state.get('company_name', 'Enterprise Partner')}\n"
        f"SENIORITY: {state.get('experience_level', 'Mid-Senior')}\n\n"
        f"JOB DESCRIPTION:\n{job_desc}\n\n"
        "Extract all programming languages, frameworks, cloud tools, "
        "domain concepts, and technical summary."
    )

    response = await structured_llm.ainvoke(
        [
            SystemMessage(content=JD_ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    parsed = JobDescriptionExtraction.model_validate(response)

    logger.info(
        "[Agent1/Extractor] Extracted: %d languages, %d frameworks, "
        "%d cloud tools, %d domain concepts",
        len(parsed.languages),
        len(parsed.frameworks),
        len(parsed.cloud_tools),
        len(parsed.domain_concepts),
    )

    return {
        "extracted_languages": parsed.languages,
        "extracted_frameworks": parsed.frameworks,
        "extracted_cloud_tools": parsed.cloud_tools,
        "extracted_domain_concepts": parsed.domain_concepts,
        "role_technical_summary": parsed.role_technical_summary,
    }


# ─── Agent 2: The Guide Architect (Generator) ───────────────────────────────


async def generator_node(state: InterviewGuideState) -> Dict[str, Any]:
    """
    Agent 2 – The Guide Architect (Generator).
    Takes Agent 1's structured technical extraction and generates the comprehensive study guide.
    Strictly forbids direct interview questions.
    """
    job_title = state.get("job_title", "Software Engineer")
    company = state.get("company_name", "Enterprise Partner")
    level = state.get("experience_level", "Mid-Senior")

    langs = ", ".join(state.get("extracted_languages", [])) or "Standard stack"
    fws = ", ".join(state.get("extracted_frameworks", [])) or "Standard frameworks"
    tools = ", ".join(state.get("extracted_cloud_tools", [])) or "Cloud & DevOps"
    concepts = ", ".join(state.get("extracted_domain_concepts", [])) or "Core CS"
    summary = state.get("role_technical_summary", "")

    logger.info(
        "[Agent2/Generator] Synthesizing study guide for '%s' at '%s'...",
        job_title,
        company,
    )

    user_prompt = f"""
TARGET POSITION: {job_title} ({level})
ORGANIZATION: {company}

AGENT 1 TECHNICAL EXTRACTION:
- Programming Languages: {langs}
- Frameworks & Runtimes: {fws}
- Cloud, DB & DevOps Tools: {tools}
- Core Domain Concepts: {concepts}
- Technical Role Summary: {summary}

FULL JOB DESCRIPTION CONTEXT:
{state.get('job_description', '')}

Generate the complete study guide:
1. role_overview_summary
2. theoretical_main_concepts (3-4 focus areas)
3. practical_implementation_guidelines (3-4 focus areas)
4. pro_tips (3-5 strategic coaching guidelines)
5. preparation_checklist (4-6 pre-interview readiness items)

Remember: STRICTLY NO DIRECT QUESTIONS OR Q&A PAIRS.
""".strip()

    structured_llm = _get_structured_llm(GuideGenerationResult.model_json_schema())

    response = await structured_llm.ainvoke(
        [
            SystemMessage(content=GUIDE_ARCHITECT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    parsed = GuideGenerationResult.model_validate(response)

    logger.info(
        "[Agent2/Generator] Generated guide: %d theory areas, %d practical areas, %d pro tips",
        len(parsed.theoretical_main_concepts),
        len(parsed.practical_implementation_guidelines),
        len(parsed.pro_tips),
    )

    return {
        "role_overview_summary": parsed.role_overview_summary,
        "theoretical_main_concepts": [
            area.model_dump() for area in parsed.theoretical_main_concepts
        ],
        "practical_implementation_guidelines": [
            area.model_dump() for area in parsed.practical_implementation_guidelines
        ],
        "pro_tips": parsed.pro_tips,
        "preparation_checklist": parsed.preparation_checklist,
    }


# ─── Agent 3: The Strict Validator (QA) ─────────────────────────────────────

_QUESTION_LEAK_REGEX = re.compile(
    r"^(what\s+is|what\s+are|how\s+do|how\s+does|how\s+can|how\s+to|why\s+is|why\s+are|explain\s+how|can\s+you)\b",
    re.IGNORECASE,
)


def _sanitize_question_leak(text: str) -> tuple[str, bool]:
    """
    Checks if a string is formatted as a direct question.
    Converts interrogative phrasing into affirmative study guidelines.
    Returns (sanitized_text, was_modified).
    """
    if not text:
        return text, False

    modified = False
    cleaned = text.strip()

    if cleaned.endswith("?"):
        cleaned = cleaned[:-1].strip()
        modified = True

    match = _QUESTION_LEAK_REGEX.match(cleaned)
    if match:
        prefix = match.group(0).lower()
        remainder = cleaned[len(prefix):].strip()
        cleaned = f"Mastery of {remainder.capitalize()}"
        modified = True

    return cleaned, modified


def _validate_focus_area(
    raw_area: Dict[str, Any],
    default_section: str,
    notes: List[str],
) -> StudyFocusArea:
    """
    Validates, normalizes, and sanitizes an individual StudyFocusArea.
    """
    area_id = raw_area.get("id")
    if not area_id or str(area_id).strip() == "":
        area_id = str(uuid.uuid4())

    section = default_section

    title = raw_area.get("title", f"{section} Topic")
    clean_title, was_mod_title = _sanitize_question_leak(title)
    if was_mod_title:
        notes.append(f"Sanitized question format in title: '{title}' -> '{clean_title}'")
    title = clean_title

    overview = raw_area.get("overview", "Core engineering competency to master.")
    clean_ov, was_mod_ov = _sanitize_question_leak(overview)
    if was_mod_ov:
        notes.append(f"Sanitized question format in overview for '{title}'")
        overview = clean_ov

    raw_concepts = raw_area.get("concepts_to_review", [])
    sanitized_concepts: List[str] = []
    for concept in raw_concepts:
        clean_c, mod_c = _sanitize_question_leak(str(concept))
        if mod_c:
            notes.append(f"Sanitized question leakage in concept item: '{concept}'")
        sanitized_concepts.append(clean_c)

    if len(sanitized_concepts) < 2:
        sanitized_concepts.extend([
            "Core theoretical foundations and mechanics",
            "Production trade-offs and performance implications",
        ])
        notes.append(f"Supplemented concept checklist for '{title}' to meet minimum depth.")

    practical_app = raw_area.get(
        "practical_application",
        "Practice whiteboarding or architecting this mechanism with real production constraints.",
    )
    clean_app, mod_app = _sanitize_question_leak(practical_app)
    if mod_app:
        notes.append(f"Sanitized question format in practical application for '{title}'")
        practical_app = clean_app

    coach_tip = raw_area.get(
        "coach_tip",
        "Lead with high-level architectural trade-offs before diving into low-level mechanics.",
    )
    clean_tip, mod_tip = _sanitize_question_leak(coach_tip)
    if mod_tip:
        notes.append(f"Sanitized question format in coach tip for '{title}'")
        coach_tip = clean_tip

    priority = raw_area.get("priority", "High Priority")
    estimated_time = raw_area.get("estimated_study_time", "35-45 mins")

    return StudyFocusArea(
        id=area_id,
        title=title,
        section=section,
        priority=priority,
        estimated_study_time=estimated_time,
        overview=overview,
        concepts_to_review=sanitized_concepts,
        practical_application=practical_app,
        coach_tip=coach_tip,
    )


def validator_node(state: InterviewGuideState) -> Dict[str, Any]:
    """
    Agent 3 – The Strict Validator (QA guardrail).
    Pure deterministic verification, schema enforcement, and Q&A leak removal.
    """
    notes: List[str] = []
    job_title = state.get("job_title", "Software Engineer")
    company = state.get("company_name", "Enterprise Partner")

    # Validate Theoretical Areas
    raw_theory = state.get("theoretical_main_concepts", [])
    validated_theory: List[Dict[str, Any]] = []
    for area in raw_theory:
        item = _validate_focus_area(area, "Key Theoretical Areas", notes)
        validated_theory.append(item.model_dump())

    if len(validated_theory) == 0:
        notes.append("Injected standard architectural baseline.")
        baseline = StudyFocusArea(
            id=str(uuid.uuid4()),
            title=f"Core Systems Architecture & Concurrency for {job_title}",
            section="Key Theoretical Areas",
            priority="Core Requirement",
            estimated_study_time="45 mins",
            overview=f"Deep review of foundational computing principles and architecture for {job_title}.",
            concepts_to_review=[
                "Concurrency and thread safety models",
                "Database isolation levels and transaction consistency",
                "Distributed system communication patterns",
            ],
            practical_application="Explain how to resolve deadlocks and race conditions in high-throughput services.",
            coach_tip="Use the STAR method to describe an incident where concurrency bugs caused production outages.",
        )
        validated_theory.append(baseline.model_dump())

    # Validate Practical Guidelines
    raw_practical = state.get("practical_implementation_guidelines", [])
    validated_practical: List[Dict[str, Any]] = []
    for area in raw_practical:
        item = _validate_focus_area(area, "Practical Implementation Focus", notes)
        validated_practical.append(item.model_dump())

    if len(validated_practical) == 0:
        notes.append("Injected standard implementation baseline.")
        baseline_pract = StudyFocusArea(
            id=str(uuid.uuid4()),
            title=f"Production Debugging & API Diagnostics for {job_title}",
            section="Practical Implementation Focus",
            priority="High Priority",
            estimated_study_time="40 mins",
            overview=f"Hands-on production troubleshooting and logging diagnostics for {company}.",
            concepts_to_review=[
                "Structured logging and distributed tracing (OpenTelemetry)",
                "Query execution plan analysis (EXPLAIN ANALYZE)",
                "Graceful degradation and circuit breaker configuration",
            ],
            practical_application="Step through diagnosing a 504 Gateway Timeout across microservice boundaries.",
            coach_tip="Articulate specific telemetry metrics (p99 latency, error budget) to demonstrate senior acumen.",
        )
        validated_practical.append(baseline_pract.model_dump())

    # Validate Pro Tips
    pro_tips = state.get("pro_tips", [])
    sanitized_tips: List[str] = []
    for tip in pro_tips:
        clean_tip, was_mod = _sanitize_question_leak(tip)
        if was_mod:
            notes.append(f"Sanitized question leakage in coach tip: '{tip}'")
        sanitized_tips.append(clean_tip)

    if len(sanitized_tips) == 0:
        notes.append("Injected fallback pro tips for interview readiness.")
        sanitized_tips = [
            f"Always Articulate Trade-Offs: When discussing architectural decisions for {job_title}, highlight constraints (latency vs consistency, rapid delivery vs maintainability).",
            "Structure Complex Explanations Top-Down: Open with a 30-second architectural summary before drilling into implementation mechanics.",
            "Anchor Theory in Real Incidents: Reference production scenarios and challenges you personally navigated.",
            f"Inquire About Scaling & Tech Debt: Ask insightful questions regarding engineering bottlenecks and deployment cadence at {company}.",
        ]

    # Validate Preparation Checklist
    checklist = state.get("preparation_checklist", [])
    sanitized_checklist: List[str] = []
    for item in checklist:
        clean_item, was_mod = _sanitize_question_leak(item)
        if was_mod:
            notes.append(f"Sanitized question leakage in checklist item: '{item}'")
        sanitized_checklist.append(clean_item)

    if len(sanitized_checklist) == 0:
        notes.append("Injected fallback preparation checklist.")
        sanitized_checklist = [
            "Review Key Theoretical Areas (System Design, Database Normalization, Distributed Communication)",
            f"Deep dive into Practical Implementation focus areas for {job_title}",
            "Prepare 2-3 concise architectural case studies from your experience detailing trade-offs",
            "Familiarize yourself with production diagnostics workflows and database query tuning",
            f"Research {company}'s core business domains and engineering values",
        ]

    role_summary = state.get("role_overview_summary")
    if not role_summary:
        role_summary = (
            f"Strategic technical preparation guide for the {job_title} position at {company}. "
            "Master the foundational theory and practical production patterns outlined below."
        )

    logger.info(
        "[Agent3/Validator] Completed QA. Validated theory: %d, practical: %d, notes: %d",
        len(validated_theory),
        len(validated_practical),
        len(notes),
    )

    return {
        "role_overview_summary": role_summary,
        "key_theoretical_areas": validated_theory,
        "practical_implementation_focus": validated_practical,
        "pro_tips": sanitized_tips,
        "preparation_checklist": sanitized_checklist,
        "validation_notes": notes,
        "is_valid": True,
    }


# ─── Graph Assembly ──────────────────────────────────────────────────────────


def build_interview_graph():
    """
    Assembles and compiles the sequential 3-agent pipeline.
    START -> extractor -> generator -> validator -> END
    """
    workflow = StateGraph(InterviewGuideState)

    workflow.add_node("extractor", extractor_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("validator", validator_node)

    workflow.add_edge(START, "extractor")
    workflow.add_edge("extractor", "generator")
    workflow.add_edge("generator", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()


interview_graph = build_interview_graph()


async def generate_interview_guide(
    job_title: str,
    job_description: str,
    experience_level: str = "Mid-Senior",
    company_name: str = "Enterprise Partner",
) -> InterviewGuideResponse:
    """
    Convenience function that invokes the interview_graph with initial state
    and returns a validated InterviewGuideResponse.
    """
    initial_state = {
        "job_title": job_title,
        "job_description": job_description,
        "experience_level": experience_level,
        "company_name": company_name,
    }

    result = await interview_graph.ainvoke(initial_state)

    theory_areas = [
        StudyFocusArea.model_validate(area)
        for area in result.get("key_theoretical_areas", [])
    ]
    practical_areas = [
        StudyFocusArea.model_validate(area)
        for area in result.get("practical_implementation_focus", [])
    ]

    return InterviewGuideResponse(
        role_overview_summary=result.get("role_overview_summary", ""),
        key_theoretical_areas=theory_areas,
        practical_implementation_focus=practical_areas,
        pro_tips=result.get("pro_tips", []),
        preparation_checklist=result.get("preparation_checklist", []),
        validation_notes=result.get("validation_notes", []),
        is_valid=result.get("is_valid", True),
    )
