from typing import List, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage, add_messages
from pydantic import BaseModel, Field

# Initial state
class AgentState(dict):
    messages: Annotated[List[BaseMessage], add_messages]
    retriever_data: str
    final_data: str
    question: str
    next: str
    explanation: str
    missing_content: str
    score: float
    tavily_search_data: str

initial_state = {
    "messages": [],
    "question": "what is cancer",
    "retriever_data": "",
    "final_data": "",
    "next": "",
    "explanation": "",
    "missing_content": "",
    "score": 0.0,
    "tavily_search_data": ""
}
