import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
import tempfile
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(page_title='Enhanced Ask the Doc App')

if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []

def add_to_sidebar(doc_name):
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def load_document(file=None, url=None):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        add_to_sidebar(file.name)
        return documents
    elif url is not None:
        loader = WebBaseLoader(url)
        documents = loader.load()
        add_to_sidebar(url)
        return documents

def generate_response(documents, openai_api_key, query_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
    db.persist()
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type='stuff',
        retriever=retriever
    )
    return qa.run(query_text)

with st.sidebar:
    st.title("Loaded Documents")
    for doc in st.session_state['document_list']:
        st.write(doc)

st.title("Document Question Answering System")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
url_input = st.text_input("Or enter a URL to load a document")

if uploaded_file or url_input:
    documents = load_document(uploaded_file, url_input)
    query_text = st.text_input("Ask a question about the document:")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    if st.button("Submit"):
        if openai_api_key and query_text:
            response = generate_response(documents, openai_api_key, query_text)
            st.write("Response:", response)
        else:
            st.warning("Please provide both an API key and a question.")
