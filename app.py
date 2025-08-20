from flask import Flask, render_template, request, redirect, url_for, session
import sys
import os

# Add backend folder to path (adjust based on your folder structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_builder import create_agent_graph
from config.config import config
from memory import memory

app = Flask(__name__)
app.secret_key = "your-very-secret-key"  # Needed for session management

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    answer = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            initial_state = {
                "messages": [],
                "question": question,
                "retriever_data": "",
                "final_data": "",
                "next": "",
                "explanation": "",
                "missing_content": "",
                "score": 0.0,
                "tavily_search_data": ""
            }
            agent = create_agent_graph(memory)
            try:
                response = agent.invoke(initial_state, config=config)
                answer = response.get("final_data", "No answer generated.")
            except Exception as e:
                answer = f"Error occurred: {str(e)}"

            # Save Q&A in session history
            chat_history = session["chat_history"]
            chat_history.append({"user": question, "assistant": answer})
            session["chat_history"] = chat_history

            return redirect(url_for("index"))

    return render_template("index.html", chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    app.run(debug=True)
