from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Issue(BaseModel):
    line: int = Field(..., ge=1)
    category: Literal["bug", "style", "security", "performance", "documentation"]
    description: str = Field(..., max_length=200)

class Action(BaseModel):
    issues: List[Issue] = Field(default_factory=list)
    final: bool = False

class Observation(BaseModel):
    code: str
    step_count: int
    previous_feedback: str = ""
    done: bool = False

class Reward(BaseModel):
    value: float
    reason: str
