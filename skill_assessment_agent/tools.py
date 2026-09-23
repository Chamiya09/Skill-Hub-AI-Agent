"""
tools.py
────────────────────────────────────────────────────────────────────────────
Dedicated, strictly-scoped job lookup tool for the Skill Assessment Agent.
Per architecture requirements:
- Read-only access restricted ONLY to the scoped job vacancy context via ASP.NET Core backend.
- Accepts a job_id (UUID).
- Returns ONLY the specific fields relevant to generating a coding assessment:
  job_title, experience_level, department, required_skills, and key_responsibilities.
- Absolutely NO direct database credentials or arbitrary database access.
"""

import os
import re
import uuid
import html
import logging
from pathlib import Path
from typing import Dict, Any, List
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)


def _clean_html(raw_html: str) -> str:
    """Utility to convert HTML rich-text to clean, readable plain text."""
    if not raw_html:
        return ""
    # Convert breaks and list items to newlines
    text = re.sub(r'<br\s*/?>', '\n', raw_html, flags=re.IGNORECASE)
    text = re.sub(r'</?(li|p|h[1-6]|tr|div)[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Unescape HTML entities (&nbsp;, &amp;, etc.) and replace non-breaking spaces
    text = html.unescape(text).replace('\xa0', ' ')
    # Normalize multiple whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def _extract_skills_and_responsibilities(clean_text: str) -> Dict[str, Any]:
    """
    Parses clean plain-text description to identify key responsibilities
    and required skills/qualifications sections if present.
    """
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    
    responsibilities: List[str] = []
    skills: List[str] = []
    current_section = "overview"

    for line in lines:
        lower_line = line.lower()
        if any(h in lower_line for h in ["responsibilities", "what you will do", "duties", "key tasks"]):
            current_section = "responsibilities"
            continue
        elif any(h in lower_line for h in ["skills", "qualifications", "requirements", "what we look for", "tech stack"]):
            current_section = "skills"
            continue
        elif any(h in lower_line for h in ["what we offer", "benefits", "about us", "role overview"]):
            current_section = "other"
            continue

        # Bullet point or content line
        clean_item = re.sub(r'^[-•*0-9.)\s]+', '', line).strip()
        if not clean_item:
            continue

        if current_section == "responsibilities":
            responsibilities.append(clean_item)
        elif current_section == "skills":
            skills.append(clean_item)

    return {
        "responsibilities": responsibilities if responsibilities else lines[:5],
        "skills": skills if skills else [line for line in lines if any(kw in line.lower() for kw in ['experience', 'proficien', 'knowledge', 'c#', 'python', 'java', 'react', 'sql', 'developer', 'engineer'])]
    }


@tool
def fetch_job_vacancy_context(job_id: str) -> Dict[str, Any]:
    """
    Fetches job vacancy requirements from the backend service for assessment question generation.
    
    Args:
        job_id: The UUID of the job vacancy to look up.
        
    Returns:
        A dictionary containing:
        - job_id: The vacancy ID.
        - job_title: The official title of the role.
        - experience_level: The required seniority level (e.g., Entry Level, Mid Level, Senior Level).
        - department: The hiring department.
        - required_skills: Extracted technical skills and qualifications.
        - key_responsibilities: Core job duties and tasks.
        - raw_description_text: Cleaned plain text of the job description for additional context.
    """
    cleaned_id = str(job_id).strip().strip("'").strip('"')
    
    # Validate UUID format to prevent malformed queries
    try:
        uuid_obj = uuid.UUID(cleaned_id)
        valid_uuid_str = str(uuid_obj)
    except (ValueError, AttributeError):
        logger.error("[Tool] Invalid UUID provided: %s", job_id)
        return {
            "error": f"Invalid job vacancy ID format: '{job_id}'. Expected a valid UUID."
        }

    backend_url = os.getenv("BACKEND_API_URL", "http://localhost:5155").rstrip("/")
    endpoint = f"{backend_url}/api/assessments/internal/job-context/{valid_uuid_str}"

    try:
        response = requests.get(endpoint, timeout=10)
        if response.status_code == 404:
            return {
                "error": f"No job vacancy found with ID: '{valid_uuid_str}'."
            }
        
        response.raise_for_status()
        data = response.json()

        rec_id = data.get("jobId") or valid_uuid_str
        title = data.get("jobTitle") or "Software Engineer"
        experience_level = data.get("experienceLevel") or "Mid Level"
        department = data.get("department") or "Engineering"
        raw_description = data.get("description") or ""

        clean_text = _clean_html(raw_description)
        extracted = _extract_skills_and_responsibilities(clean_text)

        return {
            "job_id": str(rec_id),
            "job_title": title,
            "experience_level": experience_level,
            "department": department,
            "required_skills": extracted["skills"],
            "key_responsibilities": extracted["responsibilities"],
            "raw_description_text": clean_text
        }

    except requests.exceptions.RequestException as e:
        logger.error("[Tool] HTTP request error in fetch_job_vacancy_context: %s", str(e), exc_info=True)
        return {
            "error": f"Failed to retrieve job vacancy from backend service: {str(e)}"
        }
    except Exception as e:
        logger.error("[Tool] Unexpected error in fetch_job_vacancy_context: %s", str(e), exc_info=True)
        return {
            "error": f"Failed to process job vacancy data: {str(e)}"
        }
