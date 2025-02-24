import streamlit as st
import requests
import json
from datetime import datetime

BACKEND_URL = 'http://127.0.0.1:8000/ask'

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

st.title("SahmAI")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="🙂").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🤖").write(msg["content"])

user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user", avatar="🙂").write(user_input)
    input_data = {"question": user_input, "max_retries": 3}
    
    # Show spinner while waiting for response
    with st.spinner("Thinking..."):
        try:
            response = requests.post(BACKEND_URL, json=input_data)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No answer received.")
        except Exception as e:
            answer = f"Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant", avatar="🤖").write(answer)
    st.session_state.feedback_given = False

    if not st.session_state.feedback_given:
        col1, col2 = st.columns(2)
        if col1.button("👍", key="thumbs_up"):
            st.session_state.feedback_given = True
            feedback = {
                "question": user_input,
                "answer": answer,
                "feedback": "positive",
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.setdefault("feedback", []).append(feedback)
            with open("feedback.json", "a") as f:
                f.write(json.dumps(feedback) + "\n")
            st.write("Thank you for your feedback!")
        if col2.button("👎", key="thumbs_down"):
            st.session_state.feedback_given = True
            feedback = {
                "question": user_input,
                "answer": answer,
                "feedback": "negative",
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.setdefault("feedback", []).append(feedback)
            with open("feedback.json", "a") as f:
                f.write(json.dumps(feedback) + "\n")
            st.write("Thank you for your feedback!")