import re
import litellm
from app.core.settings import settings
import os
from app.models.ec_model import ECModel
from app.models.model_catalogue import LLMModels

os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["AWS_REGION_NAME"] = settings.AWS_REGION

# messages = []
# SYSTEM_PROMPT = "You are a part of Ministry of Environment, Forest and Climate Change, Government of India. You are assisting users with their queries related to environment clearances, regulations, and policies in India. Provide accurate and concise information based on the latest guidelines and procedures."
# messages.append({"role": "system", "content": SYSTEM_PROMPT})
# USER_PROMPT = "I want to construct a residential building project in Maharashtra. These are the project details: Project Name: Green Meadows Residential Complex Description: A sustainable residential project with eco-friendly features and green spaces. Category: Residential Building Construction. Please provide the necessary steps and guidelines for obtaining environmental clearance for this project and tell which category it falls under."
# messages.append({"role": "user", "content": USER_PROMPT})

# try:
#   response = litellm.completion(
#     model="bedrock/arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
#     messages=messages,
#     response_format=ECModel
#   )
#   print(response)
# except Exception as e:
#   msg = str(e)
#   print("Error calling Bedrock/LiteLLM:", msg)
 
class LLMInterface():
    def __init__(self, session_directory, llm_api_key=None):
        self.session_directory = session_directory
        self.llm_api_key = llm_api_key
        self.model_name = LLMModels.CLAUDE_3_SONNET.value
    
    def set_model(self, model_name: str):
        self.model_name = model_name
        if model_name.startswith("bedrock/"):
            os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
            os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
            os.environ["AWS_REGION_NAME"] = settings.AWS_REGION
    
    def get_response(self, messages, response_format,) -> ECModel:
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                response_format=response_format
            )
            return response
        except Exception as e:
            msg = str(e)
            print("Error calling Bedrock/LiteLLM:", msg)
            return None
    
    async def get_response_streaming(self, messages, response_format):
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                stream=True
            )
            llm_response = {
                    'role': 'assistant',
                    'content': '',
                    'is_stream_complete': False
                }
            async for chunk in response:
                if 'choices' not in chunk or len(chunk.choices) == 0 or not chunk.choices[0].delta.content:
                    continue
                llm_response['content'] += chunk.choices[0].delta.content
                yield llm_response
            llm_response['is_stream_complete'] = True
            yield llm_response
        except Exception as e:
            msg = str(e)
            print("Error calling Bedrock/LiteLLM:", msg)
            yield None
        finally:
            return