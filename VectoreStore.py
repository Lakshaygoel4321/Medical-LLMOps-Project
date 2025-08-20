from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores.cassandra import Cassandra


import uuid
import os
import cassio

load_dotenv()

token = os.getenv("TOKEN")
end_point = os.getenv("END_POINT")

cassio.init(
    token = token,
    database_id=end_point
)


embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

vectore_store = Cassandra(
    embedding = embedding,
    session= None,
    keyspace= None,
    table_name = "cancer_expert_102"
)

# loading the data

pdf = PyPDFDirectoryLoader(path = r"C:\Users\iamda\OneDrive\Desktop\Gen ai\LLMOpS\Part-2\Medical-LLMOps-Project\Data")
loader = pdf.load()

# splitting the data
spliter = RecursiveCharacterTextSplitter(chunk_size=15000,chunk_overlap=200)
docs = spliter.split_documents(loader)

# store the data in vectore store database
vectore_store.add_documents(docs)

vectore_index = VectorStoreIndexWrapper(vectorstore = vectore_store)

# take out the retriever functionality
retriever = vectore_store.as_retriever()