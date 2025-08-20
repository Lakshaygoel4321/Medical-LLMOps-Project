from langgraph.graph import StateGraph, END
from models import AgentState
from agents.classifier import classifier_agent_function
from agents.retriever import retriever_agent_function
from agents.writer import content_writer_function
from agents.validator import validator_agent_function
from agents.web_search import tavily_search_agent_function
from agents.final_summarizer import final_summarizer_agent_function
from agents.no_answer import no_answer_function_agent

def create_agent_graph(memory):
    graph = StateGraph(AgentState)
    graph.add_node("classifier", classifier_agent_function)
    graph.add_node("retriever_agent", retriever_agent_function)
    graph.add_node("no_answer", no_answer_function_agent)
    graph.add_node("summarizer", content_writer_function)
    graph.add_node("validator", validator_agent_function)
    graph.add_node("tavily_search_tool_agent", tavily_search_agent_function)
    graph.add_node("final_summarizer", final_summarizer_agent_function)

    graph.set_entry_point("classifier")
    graph.add_edge("retriever_agent", "summarizer")
    graph.add_edge("summarizer", "validator")
    graph.add_edge("tavily_search_tool_agent", "final_summarizer")
    graph.add_edge("final_summarizer", "validator")
    graph.add_edge("no_answer", END)

    return graph.compile(checkpointer=memory)
