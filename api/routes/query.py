from fastapi import APIRouter, HTTPException

from api.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Receive a compliance question, return structured answer with citations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Deferred imports: init_observability() (api/main.py) must run before LlamaIndex loads.
    from rag.agent import build_agent
    from rag.response import parse_agent_response

    agent = build_agent()
    response = await agent.run(user_msg=request.question)
    return parse_agent_response(str(response))
