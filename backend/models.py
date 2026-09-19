from typing import Literal
from pydantic import BaseModel, Field


class BtwRouteDecision(BaseModel):
    needs_web_search: bool


class RouterDecision(BaseModel):
    datasource: str = Field(
        description="Route to 'vectorstore' if the query asks about the uploaded document, research paper, or specific file content. Route to 'web_search' for current events, or 'direct_answer' for general chit-chat."
    )


class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str


class ClaimVerificationResult(BaseModel):
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]