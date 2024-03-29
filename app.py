import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
def get_conversational_chain(model):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Sorry, I don't have information about this Question.", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    llm = ChatGroq(
            groq_api_key = groq_api_key, 
            model_name = model
    )

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, model):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model)
    
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    st.chat_message("ai").write(response["output_text"])


def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Groq💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    submit_button = st.button("Submit")
    
    st.sidebar.title("Upload PDF Files")
    model = st.sidebar.selectbox('Choose a model', ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"])

    if submit_button and user_question:
        user_input(user_question, model)

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                

if __name__ == "__main__":
    main()
