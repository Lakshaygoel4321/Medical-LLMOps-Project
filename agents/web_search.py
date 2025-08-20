from langchain_community.tools import TavilySearchResults
from models import AgentState
from langgraph.types import Command
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()



search = TavilySearchResults(max_results=4)

def tavily_search_agent_function(state: AgentState):
    """
    This function is a tavily search
    function and its work to identify
    and missing data search on the web
    """
    try:
        question = state["question"]
        missing_content = state["missing_content"]
        
        query = f"user question is: {question} And \n\n Missing data is {missing_content}"
        print("\nEntering the tavily search tool")
        print(f"Here is query: {query}")

        result = search.invoke({"query": query})

        return Command(
            update={
                "messages": state.get("messages", []) + [AIMessage(content=f"Tavily search data: {str(result)}")],
                "tavily_search_data": str(result)
            },
            goto="final_summarizer"
        )
    
    except Exception as e:
        print(f"Exception in Tavily search: {str(e)}")
        # FIXED: Return to validator with error message instead of crashing
        return Command(
            update={
                "messages": state.get("messages", []) + [AIMessage(content="Search failed, using existing content")],
                "tavily_search_data": "Search unavailable"
            },
            goto="final_summarizer"
        )