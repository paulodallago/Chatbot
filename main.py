import streamlit as st
from langchain_helper import execute_user_query

st.title("Assistente CC IFSul")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Olá! Eu sou o assistente virtual da CC IFSul. Posso ajudar?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query_text := st.chat_input("Pode perguntar!"):
    with st.chat_message("user"):
        st.markdown(query_text)

    st.session_state.messages.append({"role": "user", "content": query_text})

if query_text:
    response = execute_user_query(query_text)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})