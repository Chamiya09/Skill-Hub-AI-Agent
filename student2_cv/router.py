from fastapi import APIRouter
from pydantic import BaseModel
from student2_cv.cv_eval_graph import cv_eval_graph

router = APIRouter(prefix="/api/student2", tags=["Student 2 - CV Evaluator"])

class CVEvaluationRequest(BaseModel):
    cv_text: str
    job_description: str

@router.post("/evaluate-cv")
def evaluate_cv(request: CVEvaluationRequest):
    """
    Executes the Multi-Agent LangGraph workflow:
    1. Extractor Agent
    2. Evaluator Agent
    3. Validator Agent
    """
    # Initialize the LangGraph state with the inputs
    initial_state = {
        "cv_text": request.cv_text,
        "job_description": request.job_description
    }
    
    # Run the multi-agent workflow
    result_state = cv_eval_graph.invoke(initial_state)
    
    # Return the final JSON evaluation report (the updated state)
    return result_state
