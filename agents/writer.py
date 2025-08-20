from models import AgentState
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def content_writer_function(state: AgentState):
    """
    In this function data is already
    given we have to write the content
    properly and precisely.
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
    reference_data = state["retriever_data"]

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt),
        HumanMessage(content=f"user question: {question} And \n\n reference_data: {reference_data}")
    ])

    final_prompt = chat_prompt.format_messages(
        question=question,
        reference_data=reference_data
    )

    model = ChatGroq(model="llama-3.1-8b-instant")
    response = model.invoke(final_prompt)

    print("\n")
    print(f"Content writer generated content: {response.content}")

    return Command(
        update={
            "messages": state.get("messages", []) + [AIMessage(content=response.content)],
            "final_data": response.content
        },
        goto="validator"
    )
