from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from .scrapper import *
from dotenv import load_dotenv
import os
import hashlib
import base64

load_dotenv(dotenv_path="d:/article-chat-ai/.env")


class ChatAI:
    def __init__(self) -> None:
        self._initialize_llm()
        self._initialize_pinecone()
        self.index = self.pinecone.Index("article-embeddings")
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        self._initialize_vectore_store()

    def _initialize_llm(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash"
            )
        except Exception as e:
            print(f"Error connecting with gemini: {e}")

    def _initialize_pinecone(self):
        try:
            self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        except Exception as e:
            print(e)

    def _initialize_vectore_store(self):
        try:
            self.vectore_store = PineconeVectorStore(
                index=self.index, embedding=self.embedding
            )
        except Exception as e:
            print(e)

    def generate_id(self, url):
        url_hash = hashlib.sha256(url.encode()).digest()

        url_id = base64.urlsafe_b64encode(url_hash).decode()[:20]
        return url_id

    def create_embeddings(self, url):
        id = self.generate_id(url)
        article = self.index.fetch([id])
        print("0")
        if not article or "vectors" not in article or len(article["vectors"]) == 0:
            print("1")
            content = ScrapeInterface(url=url)
            document = [
                Document(page_content=content.scrape_and_clean(), metadata={"url": url})
            ]

            try:
                self.vectore_store.add_documents(documents=document, ids=[id])
                print(f"Document for URL {url} added with ID {id}")
            except Exception as e:
                print(f"Error adding document: {e}")
        else:
            print(f"Document for URL {url} already exists with ID {id}")

    def chat(self, prompt, chat_history):
        retriever = self.vectore_store.as_retriever()
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        system_prompt = (
            "You are an assistant for article reading. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        ai_response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        chat_history.extend(
            [HumanMessage(content=prompt), AIMessage(content=ai_response["answer"])]
        )
        return ai_response["answer"]
