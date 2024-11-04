from backend.chatai import *
import streamlit as st


class Interface:
    def __init__(self, url) -> None:
        self.url = url

    def run(self, prompt, chat_history):
        ai = ChatAI()
        id = ai.generate_id(url=self.url)
        article = ai.index.fetch([id])

        if not article or "vectors" not in article or len(article["vectors"]) == 0:
            ai.create_embeddings(self.url)

        response = ai.chat(prompt, chat_history)
        return response


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    with st.container():
        url = st.text_input("Article url", placeholder="www.skibiddi.com/article/1")
        if url:
            interface = Interface(url=url)
            prompt = st.chat_input("Enter your prompt")
            if prompt:
                st.markdown(interface.run(prompt, st.session_state.chat_history))


main()
