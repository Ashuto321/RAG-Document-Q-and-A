# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()

# ## load the GROQ And OpenAI API KEY 
# groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


# # st.image("edureka.png", width=200) 
# # st.image("character_23.jpg", width=200) 

# import streamlit as st

# # Custom CSS for centering elements
# center_css = """
# <style>
# .center {
#     display: flex;
#     justify-content: center;
#     align-items: center;
#     flex-direction: column;
# }
# </style>
# """

# # Apply the custom CSS
# st.markdown(center_css, unsafe_allow_html=True)

# # Wrap images inside a centered div
# st.markdown('<div class="center">', unsafe_allow_html=True)
# st.image("edureka.png", width=200)
# st.image("character_23.jpg", width=200)
# st.markdown('</div>', unsafe_allow_html=True)

# st.title("🗂️ Edureka Document Q&A 🤖")

# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="Llama3-8b-8192")

# prompt=ChatPromptTemplate.from_template(
# """
# Please answer the questions strictly based on the provided context.  
# Ensure the response is accurate, concise, and directly addresses the question.

# <context>
# {context}
# <context>

# Questions:  
# {input}
# """
# )

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#         st.session_state.loader=PyPDFDirectoryLoader("./w_pdf") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings





# prompt1=st.text_input(" 📚 Enter Your Question From any document")


# if st.button("Get My Embeddings"):
#     vector_embedding()
#     st.write("Edureka DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Custom CSS for centering and styling
st.markdown("""
<style>

.center {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    margin-top: 20px;
}

.stButton>button {
    color: white;
    background-color: #4CAF50; /* Green */
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    border-radius: 5px;
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #45a049;
}


.card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
            
""", unsafe_allow_html=True)

# Center and display images
st.markdown('<div class="center">', unsafe_allow_html=True)
st.image("edureka.png", width=200)
st.image("character_23.jpg", width=200)
st.markdown('</div>', unsafe_allow_html=True)

# Title with icon
st.title("🗂️ **Edureka Document Q&A 🤖**")

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Please answer the questions strictly based on the provided context.  
    Ensure the response is accurate, concise, and directly addresses the question.

    <context>
    {context}
    <context>

    Questions:  
    {input}
    """
)

# Function for embedding vectors
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./w_pdf")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )  # Splitting
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )  # Vector OpenAI embeddings

# Input field for question
prompt1 = st.text_input("📚 **Enter Your Question From Any Document**")

# Button to load embeddings
if st.button("🔄 Load Edureka DB"):
    vector_embedding()
    st.success("✅ Edureka DB is ready for queries!")

# If question is asked
if prompt1:
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start

    # Display the response
    st.markdown("### 🤖 **AI Response**")
    st.success(response['answer'])
    st.write(f"⏱️ **Response Time:** {response_time:.2f} seconds")

    # Display similar documents in an expander
    with st.expander("🔍 **Document Similarity Search Results**"):
        st.markdown("Below are the most relevant document chunks:")
        for i, doc in enumerate(response.get("context", [])):
            st.markdown(f"""
            <div class="card">
                <p>{doc.page_content}</p>
            </div>
            """, unsafe_allow_html=True)




