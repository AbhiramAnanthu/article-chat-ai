from backend.chatai import *
import streamlit as st
from urllib.parse import urlparse


class Interface:
    def __init__(self, url) -> None:
        self.url = url

    def run(self, prompt, chat_history):
        ai = ChatAI()

        vector_store = ai.handle_embeddings(self.url)

        response = ai.chat(prompt, chat_history, vector_store)
        return response


class ChatHistory:
    def __init__(self, article_name, url) -> None:
        self.chat_history = []
        self.article_name = article_name
        self.url = url

    def load_chat_history(self):
        st.write(self.chat_history)


def main():
    with st.container():
        url = st.text_input("Article url", placeholder="www.skibiddi.com/article/1")
        article_name = urlparse(url).netloc
        if article_name not in st.session_state:
            st.session_state[article_name] = ChatHistory(article_name, url)
        if url:
            interface = Interface(url=url)
            prompt = st.chat_input("Enter your prompt", key="prompt_input")
            if prompt:
                response = interface.run(
                    prompt, st.session_state[article_name].chat_history
                )
                st.markdown(response)

main()
