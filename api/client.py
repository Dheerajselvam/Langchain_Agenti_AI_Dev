import requests
import streamlit as st

def get_essay(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", 
    json = {"input":{"topic":input_text}})
    
    return response.json()["output"]

def get_poem(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", json = {"input":{"topic":input_text}})
    return response.json()["output"]

st.title("Welcome to LangServe Test")
input1 = st.text_input("Write an Essay about?")
input2 = st.text_input("Write an poem about?")

if input1:
    st.write(get_essay(input1))

if input2:
    st.write(get_poem(input2))