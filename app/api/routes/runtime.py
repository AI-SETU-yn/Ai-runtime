from fastapi import APIRouter, Depends, Request

from app.models.chat import ChatRequest, HealthResponse
from app.models.response import ChatResponse
from app.models.runtime import RuntimeContext
from app.security.auth import get_runtime_context
from app.services.chat_service import ChatService
from app.services.dependencies import get_chat_service

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status='ok')


@router.get('/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    ready_flag = getattr(request.app.state, 'is_ready', False)
    return HealthResponse(status='ok' if ready_flag else 'not_ready')


@router.post('/chat', response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    runtime_context: RuntimeContext = Depends(get_runtime_context),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await chat_service.chat(
        payload,
        runtime_context,
        request_id=request.state.request_id,
        correlation_id=request.state.correlation_id,
    )
