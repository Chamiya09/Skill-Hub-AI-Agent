"""
agent.py
────────────────────────────────────────────────────────────────────────────
Single AI Agent for Skill Assessment Question Generation.
Enforces:
1. Architecture: Exactly ONE agent with access to ONE tool (`fetch_job_vacancy_context`).
2. Calibration: Consistent moderate (Medium) difficulty without hardcoded example questions.
3. Execution Limitations: Enforces strict Judge0 sandbox constraints and library ceilings:
   - Python: Standard library + only numpy, pandas, requests, scipy, scikit-learn.
   - Other languages: Standard / built-in libraries ONLY (no external packages/NuGet/npm/Maven).
   - Headless execution (stdin/stdout, no interactive prompt loops, no network access).
4. Full test case coverage (2 sample cases, 3-5 hidden edge cases).
5. Structured Pydantic output.
"""

import os
import re
import json
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from skill_assessment_agent.tools import (
    _clean_html,
    _extract_skills_and_responsibilities,
    fetch_job_vacancy_context,
)
from skill_assessment_agent.schemas import (
    GenerateQuestionResponse,
)

# Load environment
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Technical Assessment Architect for an enterprise recruitment platform.
Your sole mission is to analyze job requisition requirements and generate a single, high-quality, practical coding assessment question.

### ARCHITECTURE & CONSTRAINTS
1. You analyze role details, required skills, and key responsibilities retrieved from the database.
2. Select the single most appropriate programming language for the assessment from the following supported languages:
   - csharp, python, javascript, typescript, java, cpp, c, go, rust, ruby, php
3. Formulate exactly ONE algorithmic/logic coding challenge tailored to the technical requirements of the role.
4. Provide a complete runnable starter code stub, a reference solution, 2 sample test cases, and 3 to 5 hidden edge-case test cases.
5. Return your final answer strictly as a valid JSON object conforming to the output schema.

### DIFFICULTY & QUESTION CALIBRATION
- The challenge must be consistently calibrated to a MODERATE (Medium) difficulty level regardless of role seniority.
- It must evaluate core problem-solving ability, algorithmic thinking, efficient data structure usage (such as arrays, strings, hash maps, sets, two pointers, stacks, queues, sliding window, or frequency counters), and clean code structure.
- DO NOT create trivia questions, esoteric language-specific riddles, ambiguous business puzzles, or competition-math problems.
- The problem description must be clear, self-contained, and professional, including problem background, input format, output format, constraints, and explicit example walkthroughs.

### EXECUTION ENVIRONMENT & JUDGE0 LIMITATIONS
The candidate's solution will be compiled and executed inside a containerized sandbox (Judge0). You MUST strictly abide by the following execution limits:
1. **Headless / Non-Interactive Execution**:
   - Code runs in a headless environment.
   - Input MUST be read from standard input (stdin) or method arguments.
   - Output MUST be printed to standard output (stdout).
   - NEVER generate code that uses interactive console prompts (e.g., `input("Enter value: ")` or `Console.Write("Enter: ")` is strictly forbidden because prompt text corrupts stdout comparison). Read raw data directly.
   - Code execution has no internet access. Do not generate questions that require external network calls.
2. **Library & Package Constraints**:
   - **Python**: You may ONLY use Python standard libraries, OR optionally one of the following 5 pre-installed packages if appropriate: `numpy`, `pandas`, `requests`, `scipy`, `scikit-learn`. Absolutely NO other third-party packages are installed. If the problem does not specifically require matrix/data analysis, prefer the standard library.
   - **All Other Languages** (`csharp`, `java`, `typescript`, `javascript`, `cpp`, `c`, `go`, `rust`, `ruby`, `php`): **STANDARD BUILT-IN LIBRARIES ONLY**. There is NO package manager (no NuGet, no npm, no Maven, no cargo, no composer). Code must compile with the language's standard SDK built-in classes and standard namespace/imports only.
3. **Language-Specific Entry Point Conventions**:
   - **Java**: The class containing `main` MUST be named `Solution` (`public class Solution { public static void main(String[] args) { ... } }`).
   - **C#**: Use `public class Solution { public static void Main(string[] args) { ... } }` with standard `using System; using System.Collections.Generic;`.
   - **C++**: Use `#include <iostream>`, `#include <vector>`, `#include <string>`, etc., with `int main() { ... return 0; }`.
   - **Go**: Use `package main`, `import "fmt"`, and `func main() { ... }`.
   - **JavaScript / TypeScript**: Node.js environment reading from `fs.readFileSync(0, 'utf-8')`.
   - **Python**: Self-contained script reading from `sys.stdin` and writing to `sys.stdout` or standard `print()`.

### TEST CASES SPECIFICATION
- **Sample Test Cases (`sampleTestCases`)**: Exactly 2 representative normal test cases visible to the candidate with clear expected output.
- **Hidden Test Cases (`hiddenTestCases`)**: 3 to 5 hidden edge cases designed to catch boundary bugs:
  - Empty input / empty string / empty collection
  - Single element / minimum valid size
  - Negative values, zeros, or opposite sign values
  - Duplicates or all-identical elements
  - Boundaries or large inputs within constraints
- Every test case `input` and `expectedOutput` must be a clean string matching the exact I/O protocol of your starter and solution code.

### OUTPUT FORMAT
Your final response MUST be a single, valid JSON object with NO preamble, NO markdown code fences (do not use ```json ... ```), and NO commentary outside the JSON. The JSON structure must match:
{
  "job_vacancy_id": "<uuid>",
  "job_title": "<title>",
  "experience_level": "<level>",
  "selected_language": "<language>",
  "question": {
    "id": "<generated_uuid>",
    "title": "<Question Title>",
    "problemStatement": "<Markdown formatted problem statement>",
    "language": "<language>",
    "difficulty": "Medium",
    "starterCode": "<Runnable starter code stub>",
    "solutionCode": "<Complete working solution code>",
    "sampleTestCases": [
      { "input": "<input_str>", "expectedOutput": "<output_str>", "isHidden": false },
      { "input": "<input_str>", "expectedOutput": "<output_str>", "isHidden": false }
    ],
    "hiddenTestCases": [
      { "input": "<input_str>", "expectedOutput": "<output_str>", "isHidden": true },
      { "input": "<input_str>", "expectedOutput": "<output_str>", "isHidden": true },
      { "input": "<input_str>", "expectedOutput": "<output_str>", "isHidden": true }
    ],
    "points": 100,
    "order": 1
  }
}
"""


def _clean_json_response(raw_text: str) -> str:
    """Extracts the outermost JSON object safely without mangling internal markdown code blocks."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return cleaned[start_idx:end_idx + 1].strip()

    return cleaned


class AssessmentAgent:
    """
    Single autonomous agent for technical assessment question generation.
    Orchestrates the tool call to fetch job requirements and generates calibrated coding challenges.
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.1):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.model_name = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")
        self.temperature = temperature

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=5000
        )

    async def generate_question(
        self,
        job_vacancy_id: str,
        focus_area: Optional[str] = None,
        difficulty: Optional[str] = "Medium",
        job_context: Optional[Dict[str, Any]] = None
    ) -> GenerateQuestionResponse:
        """
        Executes the single-agent question generation workflow:
        1. Calls the database tool to get job vacancy context.
        2. Reasons over job technical requirements and environment constraints.
        3. Generates 1 calibrated coding question with starter code and test cases.
        """
        target_diff = (difficulty or "Medium").strip().capitalize()
        if target_diff not in ["Easy", "Medium", "Hard"]:
            target_diff = "Medium"

        logger.info("[AssessmentAgent] Initiating question generation for job ID: %s (Difficulty: %s)", job_vacancy_id, target_diff)

        # The .NET backend supplies context when it has already validated the vacancy.
        if job_context:
            clean_text = _clean_html(job_context.get("description", ""))
            extracted = _extract_skills_and_responsibilities(clean_text)
            tool_data = {
                "job_id": job_vacancy_id,
                "job_title": job_context.get("job_title", "Software Engineer"),
                "experience_level": job_context.get("experience_level", "Mid Level"),
                "department": job_context.get("department", "Engineering"),
                "required_skills": extracted["skills"],
                "key_responsibilities": extracted["responsibilities"],
            }
        else:
            tool_data = fetch_job_vacancy_context.invoke({"job_id": job_vacancy_id})
        if "error" in tool_data:
            logger.error("[AssessmentAgent] Tool returned error: %s", tool_data["error"])
            raise ValueError(tool_data["error"])

        job_title = tool_data.get("job_title", "Software Engineer")
        experience_level = tool_data.get("experience_level", "Mid Level")
        department = tool_data.get("department", "Engineering")
        required_skills = tool_data.get("required_skills", [])
        key_responsibilities = tool_data.get("key_responsibilities", [])

        diff_guidelines = {
            "Easy": "The challenge must be strictly calibrated to an EASY difficulty level. Focus on fundamental logic, direct string/array operations, or straightforward condition handling with O(N) or O(1) time complexity.",
            "Medium": "The challenge must be strictly calibrated to a MODERATE (Medium) difficulty level. Focus on core problem-solving, algorithmic thinking, and efficient data structures (hash maps, two pointers, sliding window, stacks/queues).",
            "Hard": "The challenge must be strictly calibrated to an ADVANCED (Hard) difficulty level. Focus on complex algorithmic optimization, multi-step problem solving, dynamic programming, backtracking, or advanced graph/tree manipulation with strict performance constraints."
        }

        full_prompt = f"""{SYSTEM_PROMPT}

### TARGET JOB REQUISITION DATA (Retrieved from Database):
- Job Title: {job_title}
- Seniority / Experience Level: {experience_level}
- Department: {department}
- Required Skills & Qualifications: {json.dumps(required_skills, indent=2)}
- Key Responsibilities: {json.dumps(key_responsibilities, indent=2)}
- Desired Difficulty Level: {target_diff}
- Difficulty Directive: {diff_guidelines.get(target_diff, diff_guidelines['Medium'])}
"""
        if focus_area:
            full_prompt += f"\n- HR Specific Focus / Emphasis: {focus_area}\n"

        full_prompt += f"""
Now generate the single, {target_diff}-difficulty calibrated coding assessment question tailored for this requisition as a strictly valid JSON object matching the required schema with job_vacancy_id = "{job_vacancy_id}". Output ONLY the raw JSON object, no markdown fences, no explanation.
"""

        messages = [HumanMessage(content=full_prompt)]

        # Step 3: Invoke LLM with automatic fallback
        try:
            structured_llm = self.llm.with_structured_output(GenerateQuestionResponse)
            structured_response = await structured_llm.ainvoke(messages)
            if isinstance(structured_response, GenerateQuestionResponse):
                validated_response = structured_response
            else:
                validated_response = GenerateQuestionResponse.model_validate(structured_response)

            validated_response.job_vacancy_id = job_vacancy_id
            validated_response.job_title = validated_response.job_title or job_title
            validated_response.experience_level = validated_response.experience_level or experience_level
            validated_response.question.difficulty = target_diff
            validated_response.question.points = 100
            validated_response.question.order = 1
            if not validated_response.question.id:
                validated_response.question.id = f"q_{uuid.uuid4().hex[:12]}"
            logger.info("[AssessmentAgent] Successfully generated structured question '%s' in %s for role '%s'",
                        validated_response.question.title,
                        validated_response.selected_language,
                        validated_response.job_title)
            return validated_response
        except Exception as e:
            logger.warning(
                "[AssessmentAgent] Structured output with primary model '%s' failed: %s. Falling back to raw JSON parsing...",
                self.model_name,
                str(e),
            )
            try:
                ai_response = await self.llm.ainvoke(messages)
            except Exception as raw_error:
                logger.warning(
                    "[AssessmentAgent] Primary model '%s' failed: %s. Falling back to '%s'...",
                    self.model_name,
                    str(raw_error),
                    self.fallback_model,
                )
                fallback_llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name=self.fallback_model,
                    temperature=self.temperature,
                    max_tokens=2000
                )
                ai_response = await fallback_llm.ainvoke(messages)

        raw_content = ai_response.content
        logger.info("[AssessmentAgent] LLM returned content length: %d, preview: %s",
                    len(raw_content), repr(raw_content[:200]) if raw_content else "EMPTY")
        cleaned_json_str = _clean_json_response(raw_content)

        # Step 4: Validate and parse into Pydantic schema
        try:
            parsed_dict = json.loads(cleaned_json_str)

            # Ensure defaults
            if "question" in parsed_dict and isinstance(parsed_dict["question"], dict):
                if not parsed_dict["question"].get("id"):
                    parsed_dict["question"]["id"] = f"q_{uuid.uuid4().hex[:12]}"
                parsed_dict["question"]["difficulty"] = target_diff
                parsed_dict["question"]["points"] = 100
                parsed_dict["question"]["order"] = 1

            if not parsed_dict.get("job_title"):
                parsed_dict["job_title"] = job_title
            if not parsed_dict.get("experience_level"):
                parsed_dict["experience_level"] = experience_level
            if not parsed_dict.get("job_vacancy_id"):
                parsed_dict["job_vacancy_id"] = job_vacancy_id

            validated_response = GenerateQuestionResponse(**parsed_dict)
            logger.info("[AssessmentAgent] Successfully generated question '%s' in %s for role '%s'",
                        validated_response.question.title,
                        validated_response.selected_language,
                        validated_response.job_title)
            return validated_response

        except (json.JSONDecodeError, ValidationError) as err:
            logger.warning("[AssessmentAgent] First attempt JSON validation failed: %s. Attempting self-healing repair...", str(err))
            repair_prompt = (
                f"The previous output had a JSON validation error: {str(err)}.\n"
                f"Previous raw output:\n{cleaned_json_str}\n\n"
                "Please output ONLY the corrected, strictly valid JSON object conforming to the required schema."
            )
            repair_msg = await self.llm.ainvoke([
                HumanMessage(content=f"{SYSTEM_PROMPT}\n\n{repair_prompt}")
            ])
            repaired_json_str = _clean_json_response(repair_msg.content)
            repaired_dict = json.loads(repaired_json_str)
            if "question" in repaired_dict and isinstance(repaired_dict["question"], dict):
                if not repaired_dict["question"].get("id"):
                    repaired_dict["question"]["id"] = f"q_{uuid.uuid4().hex[:12]}"
                repaired_dict["question"]["difficulty"] = target_diff
                repaired_dict["question"]["points"] = 100
                repaired_dict["question"]["order"] = 1

            if not repaired_dict.get("job_title"):
                repaired_dict["job_title"] = job_title
            if not repaired_dict.get("experience_level"):
                repaired_dict["experience_level"] = experience_level
            if not repaired_dict.get("job_vacancy_id"):
                repaired_dict["job_vacancy_id"] = job_vacancy_id

            return GenerateQuestionResponse(**repaired_dict)
