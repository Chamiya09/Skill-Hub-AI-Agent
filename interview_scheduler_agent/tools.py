"""
tools.py
────────────────────────────────────────────────────────────────────────────
Backend-mediated data access tools for the Interview Scheduler Agent.
Per architecture requirements:
- Absolutely NO direct database credentials or SQL connections in Python.
- All candidate, vacancy, and calendar data is retrieved via HTTP requests to the
  ASP.NET Core backend service.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)

BACKEND_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:5155").rstrip("/")


def fetch_candidates_for_interview(job_vacancy_id: str) -> Dict[str, Any]:
    """
    Retrieves candidates selected for interview for the specified job vacancy
    from the ASP.NET Core backend.
    """
    clean_id = str(job_vacancy_id).strip()
    endpoint = f"{BACKEND_BASE_URL}/api/Events/internal/interview-candidates?jobVacancyId={clean_id}"

    try:
        response = requests.get(endpoint, timeout=10)
        if response.status_code == 404:
            return {"error": f"Job vacancy with ID '{clean_id}' was not found.", "candidates": []}

        response.raise_for_status()
        data = response.json()
        return {
            "job_vacancy_id": data.get("jobVacancyId", clean_id),
            "job_title": data.get("jobTitle", "Software Engineer"),
            "department": data.get("department", "Engineering"),
            "candidates": data.get("candidates", []),
            "total_count": data.get("totalCount", 0)
        }
    except requests.exceptions.RequestException as e:
        logger.error("[Tool] Error calling internal interview-candidates: %s", str(e), exc_info=True)
        return {
            "error": f"Failed to retrieve interview candidates from backend: {str(e)}",
            "candidates": []
        }


def fetch_blocked_slots(company_id: Optional[str], start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Retrieves existing calendar events and national holidays within the date range
    to detect scheduling clashes.
    """
    params = {
        "startDate": start_date,
        "endDate": end_date
    }
    if company_id:
        params["companyId"] = company_id

    endpoint = f"{BACKEND_BASE_URL}/api/Events/internal/existing-events"

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("blockedSlots", [])
    except requests.exceptions.RequestException as e:
        logger.warning("[Tool] Could not retrieve existing events from backend: %s. Proceeding with empty blocklist.", str(e))
        return []


def fetch_schedule_config(company_id: Optional[str]) -> Dict[str, Any]:
    """
    Retrieves company working hours and configuration.
    """
    endpoint = f"{BACKEND_BASE_URL}/api/Events/internal/schedule-config"
    params = {}
    if company_id:
        params["companyId"] = company_id

    try:
        response = requests.get(endpoint, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning("[Tool] Using fallback default schedule configuration: %s", str(e))
        return {
            "workingHoursStart": "09:00",
            "workingHoursEnd": "17:00",
            "bufferMinutes": 10,
            "timezone": "Asia/Colombo"
        }
