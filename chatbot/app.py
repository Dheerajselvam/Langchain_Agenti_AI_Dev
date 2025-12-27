from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an exceptional assistant. Please respond to the user's questions with right answer."),
            ("user", "Question:{question}")
        ]
    )

st.title('Langchain Chatbot using Ollama - moondream')
input_text = st.text_input("Ask me anything!")

llm = OllamaLLM(model="gemma3:1b")
output = StrOutputParser()
chain = prompt|llm|output

if input_text:
    st.write(chain.invoke({"question":input_text}))





