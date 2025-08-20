from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retriever_data: str
    final_data: str
    question: str
    next: str
    explanation: str
    missing_content: str
    score: float
    tavily_search_data: str




class classifier_structure(BaseModel):
    agent: Literal["retriever_agent", "no_answer"] = Field(
        ..., 
        description="If the question is related to the medical field area then you can return the name is retriever_agent or but question is not meet so return the name is no_answer"
    )


class ValidationResponse(BaseModel):
    status: Literal["correct_content", "missing_content", "incorrect_content"] = Field(
        ...,
        description="Validation status: 'correct_content', 'missing_content', or 'incorrect_content'"
    )
    
    missing_data: Optional[str] = Field(
        None,
        description="If status is 'missing_content', specify what is missing"
    )
    
    tool_usage: Literal["tool_search", "no_tool_search"] = Field(
        ...,
        description="Indicates if Tavily search tool should be used: 'tool_search' or 'no_tool_search'"
    )
    
    explanation: Optional[str] = Field(
        None,
        description="Optional explanation of the validation result"
    )
    
    score: float = Field(
        ...,
        description="Confidence score between 0.0 and 1.0"
    )