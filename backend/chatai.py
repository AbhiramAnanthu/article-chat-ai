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
from dotenv import load_dotenv
import os

path = "D:/article-chat-ai/.env"
load_dotenv(dotenv_path=path)


class LLMIntegration:
    def __init__(self):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model="models/text-embedding-004",
            )
        except Exception as e:
            print(f"Error initializing embeddings: {e}")

        try:
            self.llm = ChatGoogleGenerativeAI(
                api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash"
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")

        try:
            self.dbclient = MongoClient(os.getenv("MONGODB"))
            self.DB_NAME = "article-chat-ai"
            self.COLLECTION_NAME = "embeddings"
            self.ATLAS_VECTOR_INDEX_NAME = "article-chat-app"
            self.MONGO_DB_COLLECTION = self.dbclient[self.DB_NAME][self.COLLECTION_NAME]
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.MONGO_DB_COLLECTION,
                index_name=self.ATLAS_VECTOR_INDEX_NAME,
                embedding=self.embeddings,
                relevance_score_fn="cosine",
            )
        except Exception as e:
            print(f"Error initializing MongoDB: {e}")

    def createEmbeddings(self, path: str, tokens: int):
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                documents = [Document(page_content=text)]
                self.vector_store.add_documents(documents)
                return documents
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Error creating embeddings: {e}")

    def chat(self, input: str, chat_history: list):
        try:
            retriever = self.vector_store.as_retriever()
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
            print(response["answer"])
        except Exception as e:
            print(f"Error creating RAG chain: {e}")


integrate = LLMIntegration()
chat_history = []
while True:
    text = str(input("prompt: "))
    try:
        integrate.chat(text, chat_history)
    except Exception as e:
        print(e)
