# Skill Hub — Agentic AI Microservice

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Model-Gemini%201.5%20Pro%2FFlash-4285F4.svg)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Model-Groq%20Llama%203-F55036.svg)](https://groq.com/)
[![Azure Container Apps](https://img.shields.io/badge/Azure-Container%20Apps-0078D4.svg)](https://ca-skillhub-ai-agent-my.grayflower-f9608240.malaysiawest.azurecontainerapps.io)

An autonomous multi-agent microservice powering the intelligence layer of **Skill Hub**. Built on **FastAPI** and **LangGraph**, it coordinates semantic resume extraction, rubric-driven applicant scoring, dynamic coding assessment question generation, and intelligent interview preparation.

---

## 🌟 Component Overview & Responsibilities

- **Multi-Agent CV Evaluator:** Parses unstructured digital CVs, checks applicant qualifications against job requirements, and calculates match scores ($0\text{--}100\%$).
- **Prompt-Injection Defense:** Implements XML delimitation and input sanitization to neutralize adversarial jailbreak instructions embedded in CVs.
- **Adaptive Question Generator:** Automatically synthesizes customized technical coding problems tailored to the vacancy's seniority level.
- **Production URL:** [https://ca-skillhub-ai-agent-my.grayflower-f9608240.malaysiawest.azurecontainerapps.io](https://ca-skillhub-ai-agent-my.grayflower-f9608240.malaysiawest.azurecontainerapps.io)

---

## 🛠️ Technology Stack & Core Dependencies

- **Language:** Python 3.11
- **Web Framework:** FastAPI with Uvicorn (ASGI)
- **Agent Orchestration:** `langgraph` & `langchain-core`
- **Foundation Model Clients:**
  - `langchain-google-genai` (Google Gemini 1.5 Flash / Pro)
  - `langchain-groq` (Groq Llama 3 for low-latency inference)
- **Schema Validation:** `pydantic` v2 (Deterministic output validation)

---

## 📋 Prerequisites

- **Python:** `3.11` (recommended)
- **API Keys:** Active Google Gemini API Key and Groq API Key

---

## 🚀 Quickstart & Local Setup

### 1. Clone Repository
```bash
git clone https://github.com/Chamiya09/Skill-Hub-AI-Agent.git
cd Skill-Hub-AI-Agent
```

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Foundation Model Keys
GROQ_API_KEY=<ENTER_GROQ_API_KEY>
GEMINI_API_KEY=<ENTER_GEMINI_API_KEY>

# Model Selection
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=qwen/qwen3.8-27b

# Internal Communication
BACKEND_API_URL=<ENTER_BACKEND_API_URL>
```

### 5. Run Local Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs will be available at `http://localhost:8000/docs`.

---

## 🛡️ Security & Prompt-Injection Safeguards

This microservice incorporates defensive prompt architecture:
1. **Input Delimitation:** Raw candidate CV text is segregated inside `<candidate_untrusted_input>` boundary tags, neutralizing context overrides.
2. **Deterministic Output:** Freeform LLM markdown is rejected; models must return structured JSON validated by Pydantic models.
3. **Anonymization:** Candidate PII (names, contact details) is masked before evaluation to prevent demographic bias.

---

## ☁️ Deployment & Cloud Hosting

- **Hosting:** Hosted as an independent microservice on **Azure Container Apps** (`ca-skillhub-ai-agent-my`, Malaysia West).
- **Decoupled Compute:** Separated from the .NET core API to ensure memory-intensive token processing does not impact core transactional operations.
