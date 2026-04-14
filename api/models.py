from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(description="Employee compliance question")


class Citation(BaseModel):
    source_number: int = Field(default=0)
    doc_title: str = Field(default="")
    section: str = Field(default="")
    clause: str = Field(default="")
    clause_number: str = Field(default="")
    quote: str = Field(default="")


class Escalation(BaseModel):
    needed: bool = Field(default=False)
    reason: str = Field(default="")


class QueryResponse(BaseModel):
    answer: str = Field(default="")
    citations: list[Citation] = Field(default_factory=list)
    escalation: Escalation = Field(default_factory=Escalation)
