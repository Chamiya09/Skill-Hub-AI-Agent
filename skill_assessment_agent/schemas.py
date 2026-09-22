"""
schemas.py
────────────────────────────────────────────────────────────────────────────
Pydantic schemas and models for the Skill Assessment Agent.
Aligns directly with C# CodingQuestionItemDto and TestCaseDto contracts.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


SupportedLanguage = Literal[
    "python",
    "javascript",
    "typescript",
    "csharp",
    "java",
    "cpp",
    "c",
    "go",
    "rust",
    "ruby",
    "php"
]


class TestCaseModel(BaseModel):
    input: str = Field(
        ...,
        description="Exact raw input string passed via stdin or method call."
    )
    expectedOutput: str = Field(
        ...,
        description="Exact expected stdout output or returned string."
    )
    isHidden: bool = Field(
        default=False,
        description="Whether this test case is concealed from the candidate during testing."
    )


class GeneratedQuestionModel(BaseModel):
    id: str = Field(
        default="",
        description="Unique identifier for the question (UUID string)."
    )
    title: str = Field(
        ...,
        description="Concise, professional title of the coding challenge."
    )
    problemStatement: str = Field(
        ...,
        description="Clear problem description formatted in Markdown with input/output format, constraints, and examples."
    )
    language: SupportedLanguage = Field(
        ...,
        description="The chosen programming language best aligned with the job requirements."
    )
    difficulty: str = Field(
        default="Medium",
        description="Difficulty level of the problem, consistently calibrated to Medium."
    )
    starterCode: str = Field(
        ...,
        description="Runnable code stub provided to the candidate with input handling and function signature."
    )
    solutionCode: str = Field(
        ...,
        description="Complete reference working solution code that passes all test cases."
    )
    sampleTestCases: List[TestCaseModel] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly 2 typical sample test cases visible to the candidate."
    )
    hiddenTestCases: List[TestCaseModel] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3 to 5 hidden edge cases (e.g., empty inputs, single element, negative numbers, duplicates)."
    )
    points: int = Field(
        default=100,
        description="Total points allocated to this challenge."
    )
    order: int = Field(
        default=1,
        description="Display order."
    )


class GenerateQuestionRequest(BaseModel):
    job_vacancy_id: str = Field(
        ...,
        description="UUID of the target Job Vacancy."
    )
    focus_area: Optional[str] = Field(
        default=None,
        description="Optional HR guidance or emphasis (e.g., data manipulation, string parsing, core logic)."
    )


class GenerateQuestionResponse(BaseModel):
    job_vacancy_id: str
    job_title: str
    experience_level: str
    selected_language: str
    question: GeneratedQuestionModel

