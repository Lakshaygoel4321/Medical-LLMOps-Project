from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command
from models import AgentState, classifier_structure
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def classifier_agent_function(state: AgentState):
    """
    Classifier function it is classifying 
    next which agent is running
    """
    prompt = """
    You are a Classifier AI Agent.  
    Your task is to analyze the provided question and determine if it belongs to the **medical domain** or not.  

    Rules:  
    - If the question is related to medicine, healthcare, diseases, symptoms, diagnosis, treatments, drugs, or any medical context → return only: retriever_agent  
    - If the question is not related to the medical field → return only: no_answer  

    Important:  
    - Do not explain your decision.  
    - Do not output anything except the exact name.  

    Question: {question}
    """

    question = state["question"]
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt),
        HumanMessage(content=f"user question: {question}")
    ])

    final_prompt = chat_prompt.format_messages(question=question)

    model = ChatGroq(model="llama-3.1-8b-instant")
    llm = model.with_structured_output(classifier_structure)
    response = llm.invoke(final_prompt)

    print("--------Workflow---------")
    print(f"--------Supervisor-------{response.agent}")

    return Command(
        update={
            "messages": state.get("messages", []) + [AIMessage(content=f"classifier agent determine on which agent should work next is {response.agent}")],
            "next": response.agent
        },
        goto=response.agent
    )
