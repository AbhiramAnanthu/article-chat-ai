import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pymongo.operations import SearchIndexModel
from typing import Mapping, Any
from dotenv import load_dotenv
from .scrapper import *
import os

path = "D:/article-chat-ai/.env"
load_dotenv(dotenv_path=path)


class LLMIntegration:
    def __init__(self):
        self._initialize_components()

    def _initialize_components(self):
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_db_client()

    def _initialize_embeddings(self):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model="models/text-embedding-004",
            )
        except Exception as e:
            print(f"Error initializing embeddings: {e}")

    def _initialize_llm(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash"
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")

    def _initialize_db_client(self):
        try:
            self.dbclient = MongoClient(os.getenv("MONGODB"))
        except Exception as e:
            print(f"Error initializing MongoDB: {e}")

    def _create_search_index(self, collection, index_name):
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": 768,
                        "path": "embedding",
                        "similarity": "cosine",
                    }
                ]
            },
            name=index_name,
            type="vectorSearch",
        )
        collection.create_search_index(model=search_index_model)

    def chat(
        self, input: str, chat_history: list, vector_store: MongoDBAtlasVectorSearch
    ):
        try:
            retriever = vector_store.as_retriever()
            system_message = (
                "You are a coding assistant."
                "Use the following retrieval content to answer the user's question."
                "If you don't know the answer, say 'I am not able to answer the question.'"
                "\n\n"
                "{context}"
            )
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
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
                    ("system", system_message),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain_rag = create_stuff_documents_chain(
                self.llm, qa_prompt
            )
            rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain_rag
            )
            response = rag_chain.invoke({"input": input, "chat_history": chat_history})
            chat_history.extend(
                [HumanMessage(content=input), AIMessage(content=response["answer"])]
            )
            return response["answer"]
        except Exception as e:
            print(f"Error creating RAG chain: {e}")
