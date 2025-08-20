from models import AgentState
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END
from dotenv import load_dotenv

load_dotenv()


def no_answer_function_agent(state: AgentState):
    """
    Function to handle non-medical questions
    """
    final_data = "Sorry, I don't have the answer to this question as it's not related to medical topics."
    
    return Command(
        update={
            "messages": state.get("messages", []) + [AIMessage(content=final_data)],
            "final_data": final_data
        },
        goto=END
    )
