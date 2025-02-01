import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextInput textarea, .stChatMessage, .stChatMessage div {
        color: #ffffff !important;
        background-color: #1e1e1e;
    }
    div[data-testid="stChatMessage"] {
        background-color: #252525;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    .stChatInput input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# Piax: AI Mental Health Agent using DeepSeek")
st.caption("🤝 Always here to listen, support, and guide you.")

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    
    st.markdown("### Piax’s Capabilities")
    st.markdown("""
    - 🧘 Active Listening & Support
    - ❤️ Empathetic Conversations
    - 💡 Coping Strategies & Mindfulness
    - 📖 Personalized Guidance
    """)
    
    st.divider()
    
    st.markdown("### 🌿 Mental Health Resources")
    st.markdown("""
  #resources
    """)
    
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.7
)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are Piax, an AI mental health companion. Listen actively, respond empathetically, "
    "and provide thoughtful support. Your goal is to be a caring friend, not just an information source. "
    "Use a warm and understanding tone. Never diagnose or replace professional help."
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hey there! I'm Piax. How are you feeling today? 💙"}]

dialogue_container = st.container()

with dialogue_container:
    for message in st.session_state.message_log:
        if message["role"] == "ai":
            with st.chat_message("assistant"):
                st.markdown(message["content"], unsafe_allow_html=True)
        elif message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"], unsafe_allow_html=True)

user_input = st.chat_input("Share your thoughts...")

def generate_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_input:
    st.session_state.message_log.append({"role": "user", "content": user_input})
    
    with st.spinner("💙 Piax is listening..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    st.rerun()
