from langgraph.types import Command
from models import AgentState,ValidationResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import END,StateGraph
from dotenv import load_dotenv

load_dotenv()



def validator_agent_function(state: AgentState):
    """
    This is a validator agent and it is checking the 
    content it is write correctly or not or if not 
    correct it is generated the missing data
    """
    # FIXED: Updated prompt to match the model structure
    prompt = """
    You are a Validator AI Agent.  
    Your role is to carefully check whether the generated content is **well-structured, complete, and accurate** based on the given user question.  

    Here is your validation criteria:  
    1. **Relevance** → Does the answer fully address the user's question?  
    2. **Accuracy** → Is the information factually correct and logically sound?  
    3. **Structure** → Is the content clear, organized, and easy to follow?  
    4. **Completeness** → Is any important detail missing?  
    5. **Language Quality** → Grammar, spelling, and readability.  

    ### Instructions:  
    - If the answer is **complete, correct, and well-structured**, return:  
      - `"status": "correct_content"`  
      - `"tool_usage": "no_tool_search"`

    - If the answer is **missing some details**, return:  
      - `"status": "missing_content"`  
      - `"missing_data": <describe what is missing>`  
      - `"tool_usage": "tool_search"` (if web search needed) or `"no_tool_search"` (if internal knowledge sufficient)

    - If the answer is **incorrect or irrelevant**, return:  
      - `"status": "incorrect_content"`  
      - `"explanation": <explanation of why it's wrong>`
      - `"tool_usage": "no_tool_search"`

    - Always provide a **confidence score (0.0-1.0)** for your validation.  

    ### Inputs:  
    - **User Question:** {question}  
    - **Generated Answer:** {content}  
    """

    question = state["question"]
    content = state["final_data"]

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt),
        HumanMessage(content=f"User question: {question} \n\n And generated content: {content}")
    ])

    overall_prompt = chat_prompt.format_messages(
        question=question,
        content=content
    )

    llm = ChatGroq(model="llama-3.1-8b-instant")
    model = llm.with_structured_output(ValidationResponse)
    response = model.invoke(overall_prompt)

    status = response.status
    print("\n Entering the validator agent function")
    print(f"Status: {status}")

    if status == "correct_content":
        return Command(
            update={
                "messages": state.get("messages", []) + [AIMessage(content="Content is correct")],
                "score": response.score
            },
            goto=END
        )

    # FIXED: Handle missing_content case properly
    elif status == "missing_content":
        missing_data = response.missing_data or "Missing information not specified"
        print(f"\n Missing content: {missing_data}")
        
        explanation = response.explanation or "No explanation provided"
        print(f"\n Explanation: {explanation}")
        
        tool_usage = response.tool_usage
        print(f"Tool usage: {tool_usage}")
        
        if tool_usage == "tool_search":
            return Command(
                update={
                    "messages": state.get("messages", []) + [AIMessage(content=f"Missing content: {missing_data}")],
                    "missing_content": missing_data,
                    "explanation": explanation,
                    "score": response.score
                },
                goto="tavily_search_tool_agent"
            )
        else:
            # If no tool search needed, go back to content writer with feedback
            return Command(
                update={
                    "messages": state.get("messages", []) + [AIMessage(content=f"Content needs improvement: {missing_data}")],
                    "missing_content": missing_data,
                    "explanation": explanation,
                    "score": response.score
                },
                goto="summarizer"
            )
    
    # FIXED: Handle incorrect_content case
    else:  # incorrect_content
        explanation = response.explanation or "Content is incorrect"
        print(f"\n Content is incorrect: {explanation}")
        
        return Command(
            update={
                "messages": state.get("messages", []) + [AIMessage(content=f"Content is incorrect: {explanation}")],
                "explanation": explanation,
                "score": response.score
            },
            goto="summarizer"  # Go back to rewrite content
        )
