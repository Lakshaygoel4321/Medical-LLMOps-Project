# Medical LLMOps Question-Answering System

This repository contains a full-stack **Large Language Model Operations (LLMOps)** project designed for medical question answering. It integrates modular AI agents for classification, retrieval, content writing, validation, web search, and summarization into an orchestrated workflow. The backend is built with Python and LangChain-based agents, and a user-friendly frontend is implemented using Flask with a modern responsive UI.

---

## 🚀 Project Overview

This system allows users to ask medical and healthcare-related questions and receive professional, validated, and precisely generated answers powered by an LLMOps agent pipeline. It supports an iterative workflow with dynamic validation and web search integration to improve answer completeness and accuracy.

---

## 🗂 Folder Structure

llmops_medical_project/
│
├── agents/                 # All agent modules implementing workflow steps
│   ├── classifier.py       # Classifies question domain (medical or not)
│   ├── retriever.py        # Retrieves relevant medical knowledge
│   ├── writer.py           # Polishes and rewrites retrieved content
│   ├── validator.py        # Validates answers for completeness and correctness
│   ├── websearch.py        # Web search (Tavily) for missing content
│   ├── summarizer.py       # Final answer synthesis
│   └── noanswer.py         # Handles out-of-domain questions
│
├── frontend/               # Flask frontend app
│   └── app.py              # Flask web app with chat interface
│
├── templates/              # HTML templates for Flask frontend
│   └── index.html          # Main UI page
│
├── config.py               # Configuration variables (model names, IDs)
├── graph.py                # State graph construction and workflow orchestration
├── memory.py               # Memory saver implementation for conversation context
├── models.py               # TypedDict and Pydantic models for structured data
├── main.py                 # Main entrypoint to run the backend agent workflow standalone
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

## 💡 Features

- **Multi-agent orchestration:** Classifier, retriever, writer, validator, search, summarizer.
- **Dynamic validation:** Checks answer relevance, completeness, and triggers web search if needed.
- **Modular & extensible:** Add new agents or replace models easily.
- **Session memory:** Maintains conversation state with memory component.
- **User-friendly Flask frontend:** Clean input form, chat history, responsive design.
- **Environment-ready:** Use environment variables for API keys (e.g., Tavily Search).
- **Modern Python tooling:** Uses LangChain, Pydantic, and Groq LLM interface.

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```
git clone <your-repo-url>
cd llmops_medical_project
```

### 2. Create and activate Python environment

```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows PowerShell
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Environment variables

- Obtain your **Tavily API key** from [tavily.com](https://tavily.com) and set it:

Linux/macOS:
```
export TAVILY_API_KEY="your_api_key"
```

Windows:
```
setx TAVILY_API_KEY "your_api_key"
```

---

## 💻 Running the Backend

To test the backend LLMOps agents standalone:

```
python main.py
```

Modify the `initial_state` question in `main.py` to test custom queries.

---

## 🌐 Running the Frontend

Start the Flask web app with:

```
python frontend/app.py
```

Then open your browser at [http://localhost:5000/](http://localhost:5000/) to use the medical assistant UI.

---

## 🔧 Customization

- Switch LLM models by editing `config.py`.
- Extend agents or add new workflows in the `agents/` folder.
- Enhance frontend templates or convert to other frameworks as needed.

---

## 📝 Notes

- This demo system targets *medical domain* questions only.
- Web search via Tavily enhances answer completeness dynamically.
- Session memory persists conversation context within one browser session.
- Make sure your Python version is >= 3.8 and you have internet for model API calls.

---

## 🙏 Credits & Resources

- Built with the [LangChain](https://python.langchain.com/) framework.
- Utilizes [Pydantic](https://pydantic.dev/) for data modeling.
- Tavily Search integration for external web knowledge.
- Flask for lightweight web frontend.

---

## 📄 License

Specify your license here (e.g., MIT License).

---

Feel free to open issues or pull requests to improve the project!

---

**Enjoy building AI-powered medical assistants with modular LLMOps!** 🧬
```
