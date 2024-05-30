import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


st.title("Talk to the Doc.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192") # type: ignore

prompt = ChatPromptTemplate.from_template(
    """
    Provided the document, give most accurate response based on question.
    <document>
    {context}
    <document>
    Questions: {input}
    """
)

def vector_embedding(files):
    if "vectors" not in st.session_state:
        temp_dir = "./temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        documents = []
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # type: ignore
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

        for file in files:
            os.remove(os.path.join(temp_dir, file.name))
        os.rmdir(temp_dir)

uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
if uploaded_files:
    vector_embedding(uploaded_files)
    prompt1 = st.text_input("Ask questions.")
    
    if prompt1 and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])
