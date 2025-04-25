import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv



os.environ['GROQ_API_KEY'] = 'GROQ_API_KEY'
groq_api_key = os.getenv('GROQ_API_KEY')
if "vector" not in st.session_state:

    st.session_state.embeddings = OllamaEmbeddings(model = 'nomic-embed-text')

    st.session_state.loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("Chat with Docs - Groq Edition :) ")
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, OllamaEmbeddings(model='nomic-embed-text'))
if "vector" not in st.session_state:

    st.session_state.embeddings = OllamaEmbeddings(model = 'nomic-embed-text')
    
    # File upload section for PDFs
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Load documents
if uploaded_file is not None:
    # Load the PDF document
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Use PyPDFLoader to extract content from the uploaded PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    st.session_state.docs = docs
else:
    # Fallback to loading web-based content if no PDF is uploaded
    st.session_state.loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
    st.session_state.docs = st.session_state.loader.load()

st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)

st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
st.title("Chat with Docs - Groq Edition :) ")

llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='mixtral-8x7b-32768'
    )

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # print(doc)
            # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content)
            st.write("--------------------------------")