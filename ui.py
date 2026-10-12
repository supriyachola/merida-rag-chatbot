
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Merida Tech Minds RAG Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Merida Tech Minds")
st.subheader("Website AI Assistant")
st.write("Ask questions about our services and solutions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about Merida Tech Minds...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the website..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get(
                    "answer", "Sorry, no answer was returned."
                )
                st.markdown(answer)

                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- [{source}]({source})")

            except requests.RequestException as error:
                answer = (
                    "I couldn't connect to the chatbot API. "
                    "Please check that the API server is running."
                )
                st.error(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
