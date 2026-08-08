""" API schemas for the scoring API """

from pydantic import BaseModel

class Message(BaseModel):
    subject : str = ""
    body: str

class ScoreRequest(BaseModel):
    messages: list[Message]

class ScoreResult(BaseModel):
    spam_probability: float

class ScoreResponse(BaseModel):   
    results: list[ScoreResult]
