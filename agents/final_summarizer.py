from models import AgentState
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()




def final_summarizer_agent_function(state: AgentState):
    """
    This function properly writes the final answer
    """
    prompt = """
    You are a Content Writer Agent.  
    Your role is to generate high-quality written content in a **professional, concise, precise, and easily understandable** manner based on question.  

    Instructions:  
    - Always take the provided information from the variable `reference_data` as your source.  
    - Rewrite, restructure, and polish the content to make it clear, professional, and reader-friendly.  
    - Maintain factual accuracy from the reference data, but improve the **flow, readability, and style**.  
    - Avoid unnecessary repetition, jargon, or overly complex language.  
    - Ensure the output is engaging and structured logically.  

    Important:  
    - Only use the information in `reference_data` as your basis.  
    - Do not add unrelated or fabricated details.  

    Question: {question}
    Reference Data: {reference_data}  
    """

    question = state["question"]
    tavily_search_data = state["tavily_search_data"]
    explain = state["explanation"]
    final_data = state["final_data"]

    reference_data = f"Tavily search tool data: {tavily_search_data}\nExplanation: {explain}\nGenerated content: {final_data}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt),
        HumanMessage(content=f"User question: {question} and \n\n Reference data: {reference_data}")
    ])

    overall_prompt = chat_prompt.format_messages(
        question=question,
        reference_data=reference_data
    )

    llm = ChatGroq(model="llama-3.1-8b-instant")
    response = llm.invoke(overall_prompt)

    print("\nEntering the final summarizer agent function")
    print(f"Generated content: {response.content}") 

    return Command(
        update={
            "messages": state.get("messages", []) + [AIMessage(content=response.content)],
            "final_data": response.content
        },
        goto="validator"
    )
