from config.config import config
from memory import memory
from graph_builder import create_agent_graph


if __name__ == "__main__":

    agent = create_agent_graph(memory)

    while True:
        user_input = input("\nAsk anything about in medical field: ")

        if user_input in ["exit","clear"]:
            break

        else:
                
            initial_state = {
                "messages": [],
                "question": user_input,
                "retriever_data": "",
                "final_data": "",
                "next": "",
                "explanation": "",
                "missing_content": "",
                "score": 0.0,
                "tavily_search_data": ""
            }
            try:
                response = agent.invoke(initial_state, config=config)
                print(f"\nResponse: {response}")

                print("\n==========================================\n")
                
                data = response["final_data"]
                print(f"\nFinal Response: {data}")

            except Exception as e:
                print(f"Error executing agent: {str(e)}")


