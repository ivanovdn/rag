from fastapi import APIRouter, HTTPException

from config import settings
from api.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Receive a compliance question, return structured answer with citations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if settings.pipeline_mode == "vanilla":
        from rag.pipeline import run_query
        result = run_query(request.question)
    else:
        # Agentic mode — run LlamaIndex ReAct agent
        result = await _run_agentic(request.question)

    return result


async def _run_agentic(question: str) -> dict:
    """Run the LlamaIndex agent and parse its response."""
    from rag.agent import build_agent
    from eval.agent_wrapper import parse_agent_response

    agent = build_agent()
    response = await agent.run(user_msg=question)
    parsed = parse_agent_response(str(response))
    return parsed
