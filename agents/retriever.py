from langgraph.types import Command
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from models import AgentState
from VectoreStore import retriever

def retriever_agent_function(state:AgentState):
    """
    This function is a retriever agent function
    and it is retrieved the only medical related data
    """
    question = state["question"]
    print(f"question: -> {question}")

    try: 
        response = retriever.invoke(question)

        list_app = []
        for msg in response:
            list_app.append(msg.page_content)

        print(f"Generated Answer: {list_app[0]}")

        return Command(
            update={
                "messages": state.get("messages", []) + [AIMessage(content=list_app[0])],
                "retriever_data": list_app[0]
            },
            goto="summarizer"
        )
    
    except Exception as e:
        print(f"exception: {str(e)}")
        return Command(
            update={
                "messages": state.get("messages", []) + [AIMessage(content="Error data is not retrieved")]
            },
            goto="classifier"
        )
