# complete basic router
import json
from pyexpat.errors import messages
from fastapi import APIRouter
from app.llm.llm_interaction_wrapper import  LLMInterface
from fastapi.responses import StreamingResponse
router = APIRouter(prefix="/chat", tags=["chat"])

@router.get("/")
async def chat(user_message: str):
    llm = LLMInterface(session_directory="/tmp")

    messages = []
    SYSTEM_PROMPT = "You are a part of Ministry of Environment, Forest and Climate Change, Government of India. You are assisting users with their queries related to environment clearances, regulations, and policies in India. Provide accurate and concise information based on the latest guidelines and procedures."
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    USER_PROMPT = user_message
    messages.append({"role": "user", "content": USER_PROMPT})
    response = llm.get_response(messages, response_format=None)
    return {"response": response}

async def stream_generator(response_stream):
    async for data in response_stream:
        yield f"data: {json.dumps(data)}\n\n"

@router.get("/stream")
async def chat_stream(user_message: str):
    llm = LLMInterface(session_directory="/tmp")

    messages = []
    SYSTEM_PROMPT = "You are a part of Ministry of Environment, Forest and Climate Change, Government of India. You are assisting users with their queries related to environment clearances, regulations, and policies in India. Provide accurate and concise information based on the latest guidelines and procedures."
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    USER_PROMPT = user_message
    messages.append({"role": "user", "content": USER_PROMPT})
    
    response_stream = llm.get_response_streaming(messages, response_format=None)

    return StreamingResponse(stream_generator(response_stream), media_type="text/event-stream")
    
    