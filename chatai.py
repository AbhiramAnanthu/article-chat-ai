from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

path = ".env"
load_dotenv(dotenv_path=path)


class LLMIntegration:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model="models/text-embedding-004",
        )
        self.llm = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-pro"
        )
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

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def createEmbeddings(self, path: str, tokens: int):
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                documents = [Document(page_content=text)]
                self.vector_store.add_documents(documents)
                return documents
        except FileNotFoundError as e:
            print(e)

    def createRAG(self):
        retriever = self.vector_store.as_retriever()
        system_message = (
            "You are coding assistant."
            "Use the following retrieval content to answer to the prompt."
            "If you don't know the answer. So i am not able to answer the question"
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", "{input}")]
        )
        question_answer_chain_rag = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain_rag)
        response = rag_chain.invoke(
            {
                "input": "What kind of neural network is implemented here. I mean what is the example show here ?"
            }
        )
        print(response["answer"])


integrate = LLMIntegration()
integrate.createRAG()
