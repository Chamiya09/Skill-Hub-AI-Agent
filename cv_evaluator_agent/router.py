from fastapi import APIRouter
from pydantic import BaseModel
from cv_evaluator_agent.cv_eval_graph import cv_eval_graph

router = APIRouter(tags=["CV Evaluator Agent"])


class CVEvaluationRequest(BaseModel):
    cv_text: str
    job_description: str
    required_skills: str


@router.post("/api/student2/evaluate-cv")
@router.post("/api/cv-evaluator/evaluate-cv")
async def evaluate_cv(request: CVEvaluationRequest):
    """
    Executes the Multi-Agent LangGraph workflow using Groq LLM:
    1. Extractor Agent
    2. Evaluator Agent
    3. Validator Agent
    """
    initial_state = {
        "cv_text": request.cv_text,
        "job_description": f"{request.job_description}\nRequired Skills: {request.required_skills}"
    }

    # Run the multi-agent workflow
    result_state = await cv_eval_graph.ainvoke(initial_state)

    # Ensure match_score reflects the validated final score
    if "final_match_score" in result_state and result_state["final_match_score"] is not None:
        result_state["match_score"] = int(result_state["final_match_score"])

    # Ensure list fields default to empty lists if None
    result_state.setdefault("strengths", [])
    result_state.setdefault("missing_skills", [])
    result_state.setdefault("validation_notes", [])
    result_state.setdefault("recommendation", "Evaluation completed successfully.")

    # Returns the state dict directly.
    # C# captures: match_score, strengths, missing_skills, recommendation, validation_notes
    return result_state
