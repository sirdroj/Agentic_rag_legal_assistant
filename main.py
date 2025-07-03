import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# Import the law assistant
from agentic_rag_law_assistant import AgenticRAGLawAssistant, SearchResult

# Page configuration
st.set_page_config(
    page_title="Indian Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
}
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #2a5298;
}
.user-message {
    background-color: #e3f2fd;
    border-left-color: #1976d2;
}
.assistant-message {
    background-color: #f3e5f5;
    border-left-color: #7b1fa2;
}
.source-box {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.doc-box {
    background-color: #f5f5f5;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    max-height: 200px;
    overflow-y: auto;
}
.metric-box {
    background-color: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'law_assistant' not in st.session_state:
    st.session_state.law_assistant = None
if 'assistant_initialized' not in st.session_state:
    st.session_state.assistant_initialized = False

def initialize_assistant(api_key: str, model_name: str = "llama3-8b-8192") -> bool:
    """Initialize the law assistant with Groq API key"""
    try:
        st.session_state.law_assistant = AgenticRAGLawAssistant(
            groq_api_key=api_key,
            groq_model=model_name,
            embedding_model="all-MiniLM-L6-v2",
            reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        st.session_state.assistant_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize assistant: {str(e)}")
        return False

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message"""
    message_class = "user-message" if is_user else "assistant-message"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{'You' if is_user else 'Law Assistant'}:</strong><br>
            {message.get('content', message.get('response', ''))}
        </div>
        """, unsafe_allow_html=True)

def display_sources(sources: List[str]):
    """Display information sources"""
    if sources:
        st.markdown("### 📚 Information Sources")
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-box">
                <strong>Source {i}:</strong> {source}
            </div>
            """, unsafe_allow_html=True)

def display_retrieved_docs(docs: List[str], vector_results: List[SearchResult] = None, web_results: List[SearchResult] = None):
    """Display retrieved documents with details"""
    if docs:
        st.markdown("### 📄 Retrieved Documents")
        
        # Vector search results
        if vector_results:
            st.markdown("#### 🔍 Vector Database Results")
            for i, result in enumerate(vector_results, 1):
                with st.expander(f"Vector Result {i} (Score: {result.relevance_score:.3f})"):
                    st.markdown(f"""
                    <div class="doc-box">
                        <strong>Content:</strong><br>
                        {result.content[:500]}{'...' if len(result.content) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result.metadata:
                        st.json(result.metadata)
        
        # Web search results  
        if web_results:
            st.markdown("#### 🌐 Web Search Results")
            for i, result in enumerate(web_results, 1):
                with st.expander(f"Web Result {i}"):
                    st.markdown(f"""
                    <div class="doc-box">
                        <strong>Content:</strong><br>
                        {result.content[:500]}{'...' if len(result.content) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result.metadata:
                        st.markdown(f"**Title:** {result.metadata.get('title', 'N/A')}")
                        st.markdown(f"**URL:** {result.metadata.get('href', 'N/A')}")

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Indian Law Assistant</h1>
        <p>Your AI-powered legal research companion for Indian law (Powered by Groq)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key to use the assistant. Get one free at https://console.groq.com"
        )
        
        # Model selection
        model_options = {
            "llama3-8b-8192": "Llama 3 8B (Fast, Good for most queries)",
            "llama3-70b-8192": "Llama 3 70B (Slower, Better quality)",
            "mixtral-8x7b-32768": "Mixtral 8x7B (Balanced performance)",
            "gemma-7b-it": "Gemma 7B (Google's model)"
        }
        
        selected_model = st.selectbox(
            "Select Groq Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="Choose the LLM model for generating responses"
        )
        
        if api_key and not st.session_state.assistant_initialized:
            if st.button("Initialize Assistant"):
                with st.spinner("Initializing Law Assistant..."):
                    if initialize_assistant(api_key, selected_model):
                        st.success("Assistant initialized successfully!")
                        st.info(f"Using model: {model_options[selected_model]}")
                    else:
                        st.error("Failed to initialize assistant")
        
        # Settings
        st.header("📊 Settings")
        
        if st.session_state.assistant_initialized:
            st.success("✅ Assistant Ready")
            
            # Display current settings
            st.markdown("**Current Configuration:**")
            st.write("- Vector Database: ✅ Loaded")
            st.write("- Web Search: ✅ Enabled")
            st.write("- Reranking: ✅ Enabled")
        else:
            st.warning("⚠️ Assistant not initialized")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                display_chat_message(message, is_user=True)
            else:
                display_chat_message(message, is_user=False)
        
        # Chat input
        if st.session_state.assistant_initialized:
            user_query = st.text_input(
                "Ask a question about Indian law:",
                placeholder="e.g., What are the consumer rights under Indian law?",
                key="user_input"
            )
            
            col_send, col_examples = st.columns([1, 2])
            
            with col_send:
                if st.button("Send", type="primary"):
                    if user_query:
                        # Add user message to history
                        st.session_state.chat_history.append({
                            'type': 'user',
                            'content': user_query,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Get response from assistant
                        with st.spinner("Thinking..."):
                            response = st.session_state.law_assistant.ask_question(user_query)
                            
                            # Add assistant response to history
                            st.session_state.chat_history.append({
                                'type': 'assistant',
                                'content': response['response'],
                                'sources': response['sources'],
                                'retrieved_docs': response['retrieved_docs'],
                                'vector_results': response['vector_results'],
                                'web_results': response['web_results'],
                                'timestamp': response['timestamp']
                            })
                        
                        st.rerun()
            
            with col_examples:
                st.markdown("**Example Questions:**")
                example_questions = [
                    "What are the rights of consumers under Indian law?",
                    "What is the procedure for filing a PIL?",
                    "What are the fundamental rights in Indian Constitution?",
                    "What is the process for property registration in India?"
                ]
                
                for question in example_questions:
                    if st.button(f"📝 {question[:30]}...", key=f"example_{hash(question)}"):
                        st.session_state.chat_history.append({
                            'type': 'user',
                            'content': question,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        with st.spinner("Thinking..."):
                            response = st.session_state.law_assistant.ask_question(question)
                            
                            st.session_state.chat_history.append({
                                'type': 'assistant',
                                'content': response['response'],
                                'sources': response['sources'],
                                'retrieved_docs': response['retrieved_docs'],
                                'vector_results': response['vector_results'],
                                'web_results': response['web_results'],
                                'timestamp': response['timestamp']
                            })
                        
                        st.rerun()
        
        else:
            st.info("Please initialize the assistant first by entering your Groq API key in the sidebar.")
    
    with col2:
        st.header("📊 Information Panel")
        
        # Display sources and documents for the last assistant message
        if st.session_state.chat_history:
            last_message = None
            for message in reversed(st.session_state.chat_history):
                if message['type'] == 'assistant':
                    last_message = message
                    break
            
            if last_message:
                # Metrics
                st.markdown("### 📈 Query Metrics")
                
                num_vector_results = len(last_message.get('vector_results', []))
                num_web_results = len(last_message.get('web_results', []))
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{num_vector_results}</h3>
                        <p>Vector Results</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{num_web_results}</h3>
                        <p>Web Results</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display sources
                display_sources(last_message.get('sources', []))
                
                # Display retrieved documents
                display_retrieved_docs(
                    last_message.get('retrieved_docs', []),
                    last_message.get('vector_results', []),
                    last_message.get('web_results', [])
                )
        
        else:
            st.info("Ask a question to see information sources and retrieved documents here.")

if __name__ == "__main__":
    main()