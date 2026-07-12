from pydantic import BaseModel, Field
from typing import List

class UserQuestion(BaseModel):
    question: str = Field(..., min_length=3)

class SearchResult(BaseModel):
    score: float
    text: str

class RAGResponse(BaseModel):
    answer: str
    contexts: List[str]