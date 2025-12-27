from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from fastapi import FastAPI
from langserve import add_routes

import streamlit as st
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm1 = OllamaLLM(model="gemma3")
llm2 = OllamaLLM(model="gemma3:1b")


prompt1 = ChatPromptTemplate.from_template("Write a 100 words essay on {topic}. It should be educational.")
prompt2 = ChatPromptTemplate.from_template("Write a poem on {topic}, for a 5 year old to sing.")

chain1 = prompt1|llm1
chain2 = prompt2|llm2

app = FastAPI(
    title = "LangChain Server",
    version = "1.0",
    description="A server for essay and poems"
    )

add_routes(
        app,
        chain1,
        path="/essay"
    )

add_routes(
        app,
        chain2,
        path="/poem"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port = 8000)




