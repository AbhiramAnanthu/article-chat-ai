from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
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
                start_token = 0
                end_token = tokens // 3
                documents = []
                while start_token < tokens:
                    documents.append(Document(page_content=text[start_token:end_token]))
                    start_token = end_token
                    end_token += tokens // 3
                self.vector_store.add_documents(documents)
                return documents
        except FileNotFoundError as e:
            print(e)

    def createRAG(self):
        # retriever = self.vector_store.as_retriever()
        # prompt = PromptTemplate.from_template("{text}")
        # embeddings = self.createEmbeddings("article.txt", 3394)
        # #context =
        # #formatted_context = self.format_docs(context)
        # messages = [
        #     ("system", "Use the provided documents to answer the following question."),
        #     ("context", formatted_context),
        #     ("human", "what kind of neural network is implemented here"),
        # ]
        # ai_msg = self.llm.invoke(messages)
        # print(ai_msg)


integrate = LLMIntegration()
integrate.createRAG()
