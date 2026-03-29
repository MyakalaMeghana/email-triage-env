from pydantic import BaseModel
from typing import List

class Observation(BaseModel):
    emails: List[str]
    remaining: int

class Action(BaseModel):
    action_type: str   # "read", "delete", "reply"

class Reward(BaseModel):
    score: float
