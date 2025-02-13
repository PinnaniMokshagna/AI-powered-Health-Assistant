import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

# ✅ Set page config with custom colors and icons
st.set_page_config(
    page_title="Medical Chatbot",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a Medical Chatbot designed to provide health-related information."
    }
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage.user {
        background-color: #e3f2fd;
    }
    .stChatMessage.assistant {
        background-color: #f5f5f5;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = "pcsk_3tvNku_A9UeFeqpn3DqSuCKf1RJ8rpnDdJdm69bXowNF3pEwsUmbvfZZTLnVDdEMbsqWJE"
GROQ_API_KEY = "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"

# ✅ Cache embeddings to prevent reloading on every interaction (silent loading)
@st.cache_resource
def load_embeddings():
    return download_hugging_face_embeddings()

# ✅ Cache Pinecone retriever (silent loading)
@st.cache_resource
def load_retriever():
    embeddings = load_embeddings()
    index_name = "medicalbot"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Cache LLM model (silent loading)
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0.4,
        max_tokens=500,
        model_name="llama-3.3-70b-versatile"
    )

# ✅ Track initialization state
if "initialized" not in st.session_state:
    st.session_state["retriever"] = load_retriever()
    st.session_state["llm"] = load_llm()
    st.session_state["initialized"] = True

# Create RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(st.session_state["llm"], prompt)
rag_chain = create_retrieval_chain(st.session_state["retriever"], question_answer_chain)

# Sidebar Navigation with Dropdown
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Chat", "About"], index=0)

if page == "Chat":
    st.title("🩺 AI Powered Health  Assistant")
    st.write("Ask me anything about health!")

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"], avatar="⚕️" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            bot_response = response["answer"]

        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
        
        with st.chat_message("assistant", avatar="⚕️"):
            st.markdown(bot_response)

elif page == "About":
    st.title("ℹ About the Medical Chatbot")

    # Custom CSS for cards and styling
    st.markdown("""
    <style>
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card h3 {
        color: #2c3e50;
        margin-top: 0;
    }
    .card p {
        color: #34495e;
    }
    .icon {
        font-size: 24px;
        margin-right: 10px;
    }
    .feature-list {
        margin-left: 20px;
    }
    .feature-list li {
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Overview Card
    st.markdown("""
    <div class="card">
        <h3>📌 Overview</h3>
        <p>
            The <strong>Medical Chatbot</strong> is an AI-powered virtual assistant designed to provide instant, accurate, and reliable responses to health-related inquiries. 
            By leveraging cutting-edge technologies like <strong>Natural Language Processing (NLP)</strong> and <strong>Retrieval-Augmented Generation (RAG)</strong>, 
            this chatbot bridges the gap between users and medical knowledge, offering a seamless and interactive experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack Card
    st.markdown("""
    <div class="card">
        <h3>🛠 Technology Stack</h3>
        <p>
            The chatbot is built using a robust and scalable technology stack:
        </p>
        <ul class="feature-list">
            <li><span class="icon">⚕️</span> <strong>Language Model:</strong> Groq’s <strong>LLama-3.3-70B Versatile</strong> for advanced natural language understanding and generation.</li>
            <li><span class="icon">🔍</span> <strong>Retrieval System:</strong> Pinecone <strong>Vector Database</strong> for efficient document storage and similarity-based search.</li>
            <li><span class="icon">🧠</span> <strong>Frameworks:</strong> <strong>LangChain</strong> for orchestrating AI-driven workflows and response generation.</li>
            <li><span class="icon">🎨</span> <strong>Frontend:</strong> <strong>Streamlit</strong> for an intuitive and interactive user interface.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Features Card
    st.markdown("""
    <div class="card">
        <h3>✨ Features</h3>
        <p>
            The Medical Chatbot is packed with powerful features to enhance user experience:
        </p>
        <ul class="feature-list">
            <li><span class="icon">💡</span> <strong>Intelligent Health Query Resolution:</strong> Retrieves and generates relevant medical insights in real-time.</li>
            <li><span class="icon">📚</span> <strong>Conversational Memory:</strong> Stores chat history for easy reference and continuity.</li>
            <li><span class="icon">⚡</span> <strong>Efficient Search Mechanism:</strong> Utilizes embeddings for fast and accurate document retrieval.</li>
            <li><span class="icon">🔒</span> <strong>Secure & Scalable:</strong> Built with <strong>Pinecone</strong> and <strong>Groq LLM</strong> to handle large-scale queries securely.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Card
    st.markdown("""
    <div class="card">
        <h3>🔧 How It Works</h3>
        <p>
            The chatbot follows a streamlined process to deliver accurate and contextually relevant responses:
        </p>
        <ol class="feature-list">
            <li><strong>User Query Processing:</strong> The chatbot receives a user's health-related question.</li>
            <li><strong>Retrieval Mechanism:</strong> Relevant medical documents are retrieved using <strong>Pinecone Vector Store</strong>.</li>
            <li><strong>AI-Powered Response:</strong> The chatbot generates responses by combining retrieved knowledge with <strong>LLM-based inference</strong>.</li>
            <li><strong>Interactive Experience:</strong> Users receive instant and contextually accurate answers.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer Card
    st.markdown("""
    <div class="card">
        <h3>⚠ Disclaimer</h3>
        <p>
            <strong>This chatbot is intended for informational purposes only.</strong> It does not provide professional medical advice, diagnosis, or treatment. 
            Always consult a qualified healthcare provider for medical concerns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Developer Contact Card
    st.markdown("""
    <div class="card">
        <h3>👩‍💻 Developer Contact</h3>
        <p>
            <strong>Developed by Ananya Krishna</strong><br>
            📧 Email: <a href="mailto:pinnanimokshagna@gmail.com">pinnanimokshagna@gmail.com</a><br>
            🔗 GitHub: <a href="https://github.com/PinnaniMokshagna" target="_blank">https://github.com/PinnaniMokshagna</a><br>
            🔗 LinkedIn: <a href="https://www.linkedin.com/in/mokshagna-varma-ab5508281" target="_blank">https://www.linkedin.com/in/mokshagna-varma-ab5508281/</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
