from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Setup Groq LLM (Make sure GROQ_API_KEY is in .env)
llm = ChatGroq(model_name="llama3-70b-8192")

# System prompt to specialize the assistant
system_message = SystemMessage(content="""
You are a QA (Quality Assurance) assistant.

✅ You MUST:
- Answer ONLY questions related to QA, software testing, test automation, test strategy, tools like Selenium, Pytest, JUnit, TestNG, Playwright, etc.
- Help users write test cases, debug test scripts, or understand QA concepts.

❌ You MUST NOT:
- Answer questions outside the QA/testing domain (e.g., general programming, medical, finance, history, etc.)
- If the question is out of scope, respond with:
  "❗Sorry, I am a QA assistant and can only help with software testing-related topics."

Be concise, give examples where needed, and act like a senior QA engineer guiding a junior.
""")

# Main chat handler


def chat(user_input, history):
    langchain_history = [system_message]

    # Convert Gradio-style message history to LangChain format
    for message in history:
        role = message["role"]
        content = message["content"]
        if role == "user":
            langchain_history.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_history.append(AIMessage(content=content))

    langchain_history.append(HumanMessage(content=user_input))
    response = llm.invoke(langchain_history)

    # Return just the assistant's message; Gradio manages history
    return response.content


# Gradio Chat Interface with OpenAI-style messages
chat_interface = gr.ChatInterface(
    fn=chat,
    title="QAgent - QA Chatbot",
    description="Chat with QAgent on QA assistance!",
    theme="default",
    type="messages"  # Important for modern Gradio versions
)

# Launch app
if __name__ == "__main__":
    chat_interface.launch()
