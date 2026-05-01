import streamlit as st
import os
from brain import process_document, get_answer

st.title("📄 DocScanner Chatbot")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.vector_db is None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("Reading document..."):
        st.session_state.vector_db = process_document("temp.pdf")
    st.success("Document Ready!")

# Chat Interface
if prompt := st.chat_input("Ask about your file"):
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        response = st.write_stream(get_answer(prompt, st.session_state.vector_db))