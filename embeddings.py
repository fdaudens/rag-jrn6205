from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.docstore.document import Document

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Charger .env et API key
load_dotenv(find_dotenv(".streamlit/secrets.toml"))

openai_api_key = st.secrets.OPENAI_API_KEY

@st.cache_resource(show_spinner=False)
def initialize_chain(system_prompt, _memory):
    llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-4-1106-preview", streaming=True)

    # documents = []
    # for file in os.listdir("data/notes-de-cours/"):
    #     if file.endswith(".pdf"):
    #         pdf_path = "./data/notes-de-cours/" + file
    #         loader = PyPDFLoader(pdf_path)
    #         documents.extend(loader.load())
    #     elif file.endswith('.docx') or file.endswith('.doc'):
    #         doc_path = "./data/notes-de-cours/" + file
    #         loader = Docx2txtLoader(doc_path)
    #         documents.extend(loader.load())
    #     elif file.endswith('.txt'):
    #         text_path = "./data/notes-de-cours/" + file
    #         loader = TextLoader(text_path)
    #         documents.extend(loader.load())

    documents = []
    folder_path = "./data"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".md"):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(page_content=content, metadata={}))
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(chunked_documents, embedding=embeddings, persist_directory="./vector")
        vectorstore.persist()

        retriever = vectorstore.as_retriever(
            search_type="mmr", #mmr 
            search_kwargs={"k": 6},
        )    

        qa = ConversationalRetrievalChain.from_llm(
            llm, 
            retriever=retriever,
            memory=_memory, 
            verbose = True) #return_source_documents=True
            
        return qa
