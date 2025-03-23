import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
import os
from dotenv import load_dotenv
import time
import json
from datetime import datetime
# Import web scraping libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import re
import pickle
import hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Define constants
MODEL_OPTIONS = {
    "llama3": "Llama 3 (Default)",
    "phi3": "Phi-3 (Microsoft)",
    "mistral": "Mistral (Fast)",
    "gemma": "Gemma (Google)",
    "mixtral": "Mixtral (Powerful)",
    "neural-chat": "Neural Chat (Intel)"
}

DEFAULT_SYSTEM_PROMPT = """You are an advanced AI research assistant with access to various search tools including web search, academic papers, Wikipedia, and web scraping.
Your goal is to provide comprehensive, accurate, and helpful answers.
Always verify information through multiple sources when possible.
Format your responses with clear headings, bullet points for key information, and organized sections.
When you need detailed information from a specific website, use the WebScraper tool.
Cite your sources at the end of your responses.
If you're uncertain about information, clearly state the limitations of your knowledge."""

# Streamlit app setup
st.set_page_config(page_title="Advanced Research Assistant", layout="wide")
st.title("🔬 Advanced Research Assistant")
st.markdown("""
    An advanced AI research assistant equipped with web search, academic research, web scraping capabilities, and seamless knowledge base integration.
    It provides intelligent answers with enhanced data retrieval and analysis using locally-run open-source models.
""")

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #306998;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #4B8BBE;
        color: white;
        border-radius: 5px;
    }
    .tool-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .thinking-process {
        background-color: #f8f9fa;
        border-left: 4px solid #4B8BBE;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.9rem;
    }
    .source-citation {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 1rem;
        border-top: 1px solid #dee2e6;
        padding-top: 0.5rem;
    }
    .stAlert {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm an advanced search assistant that can search the web, scrape websites, access academic papers, and knowledge bases. I run locally using open-source models. How can I help you today?"}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "settings" not in st.session_state:
    st.session_state.settings = {
        "model": "llama3:8b",
        "temperature": 0.7,
        "show_thinking": True,
        "tools_enabled": {
            "duckduckgo": True,
            "arxiv": True,
            "wikipedia": True,
            "web_scraper": True,
            "vector_store": False
        },
        "max_iterations": 5,
        "max_tokens": 4096,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "memory_messages": 8,
        "local_server": "http://localhost:11434"
    }

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}

# Utility functions for vector storage
def get_embeddings_model():
    """Initialize and return the embeddings model"""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def create_vector_store(texts, source):
    """Create a vector store from texts"""
    embeddings = get_embeddings_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([texts])
    for doc in docs:
        doc.metadata["source"] = source
    
    return FAISS.from_documents(docs, embeddings)

def get_cache_key(url):
    """Generate a cache key for a URL"""
    return hashlib.md5(url.encode()).hexdigest()

def save_vector_store(vs, key):
    """Save vector store to disk"""
    cache_dir = ".vector_cache"
    os.makedirs(cache_dir, exist_ok=True)
    with open(f"{cache_dir}/{key}.pkl", "wb") as f:
        pickle.dump(vs, f)

def load_vector_store(key):
    """Load vector store from disk"""
    cache_dir = ".vector_cache"
    try:
        with open(f"{cache_dir}/{key}.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

# Advanced web scraping function with caching
def scrape_website(url: str) -> str:
    """Scrape and extract the content from a specific URL with caching."""
    try:
        # Clean and validate URL
        url = url.strip()
        if not re.match(r'^https?://', url):
            url = 'https://' + url
            
        # Check cache first
        cache_key = get_cache_key(url)
        if cache_key in st.session_state.vector_stores:
            return f"Retrieved content from {url} (cached). Use 'vector_search' to query this content."
        
        # Load from disk cache
        disk_cache = load_vector_store(cache_key)
        if disk_cache:
            st.session_state.vector_stores[cache_key] = {
                "store": disk_cache,
                "url": url,
                "timestamp": datetime.now()
            }
            return f"Retrieved content from {url} (disk cache). Use 'vector_search' to query this content."
        
        # Perform actual scraping
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = "\n\n".join([doc.page_content for doc in docs])
        
        # Create and store the vector store
        vs = create_vector_store(content, url)
        st.session_state.vector_stores[cache_key] = {
            "store": vs,
            "url": url,
            "timestamp": datetime.now()
        }
        save_vector_store(vs, cache_key)
        
        # Return a preview of the content
        preview = content[:1000] + "..." if len(content) > 1000 else content
        return f"Successfully scraped {url}. Content preview:\n\n{preview}\n\nUse 'vector_search' to query this content."
    except Exception as e:
        return f"Error scraping the website: {str(e)}"

def vector_search(query: str) -> str:
    """Search through the vector stores for relevant information."""
    if not st.session_state.vector_stores:
        return "No websites have been scraped yet. Use the WebScraper tool first."
    
    results = []
    for key, vs_data in st.session_state.vector_stores.items():
        try:
            docs = vs_data["store"].similarity_search(query, k=2)
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": vs_data["url"],
                    "score": doc.metadata.get("score", 1.0)  # Default score if not available
                })
        except Exception as e:
            results.append({
                "content": f"Error searching vector store: {str(e)}",
                "source": vs_data["url"],
                "score": 0
            })
    
    # Sort results by relevance
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Format the results
    output = "### Vector Search Results\n\n"
    for i, result in enumerate(results[:5]):  # Top 5 results
        output += f"**Result {i+1} from {result['source']}**\n{result['content']}\n\n"
    
    return output

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # Model Settings
    st.subheader("Model Settings")
    
    # Ollama server URL
    ollama_server = st.text_input(
        "Ollama Server URL",
        value=st.session_state.settings["local_server"],
        help="URL of your local Ollama server. Default is http://localhost:11434"
    )
    st.session_state.settings["local_server"] = ollama_server
    
    # Model selection
    selected_model = st.selectbox(
        "Choose a model",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS[x],
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.settings["model"])
    )
    st.session_state.settings["model"] = selected_model
    
    # Check model status
    model_status = st.empty()
    
    # Temperature
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings["temperature"],
        step=0.1
    )
    st.session_state.settings["temperature"] = temperature
    
    # Maximum tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=512,
        max_value=8192,
        value=st.session_state.settings["max_tokens"],
        step=512
    )
    st.session_state.settings["max_tokens"] = max_tokens
    
    # System prompt
    with st.expander("Customize System Prompt"):
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.settings["system_prompt"],
            height=200
        )
        st.session_state.settings["system_prompt"] = system_prompt
        
        if st.button("Reset to Default"):
            st.session_state.settings["system_prompt"] = DEFAULT_SYSTEM_PROMPT
            st.rerun()
    
    # Tool settings
    st.subheader("Search Tools")
    
    # Toggle for each tool
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.settings["tools_enabled"]["duckduckgo"] = st.checkbox(
            "Web Search",
            value=st.session_state.settings["tools_enabled"]["duckduckgo"]
        )
        st.session_state.settings["tools_enabled"]["arxiv"] = st.checkbox(
            "Academic Papers (arXiv)",
            value=st.session_state.settings["tools_enabled"]["arxiv"]
        )
    with col2:
        st.session_state.settings["tools_enabled"]["wikipedia"] = st.checkbox(
            "Wikipedia",
            value=st.session_state.settings["tools_enabled"]["wikipedia"]
        )
        st.session_state.settings["tools_enabled"]["web_scraper"] = st.checkbox(
            "Web Scraper",
            value=st.session_state.settings["tools_enabled"]["web_scraper"]
        )
    
    st.session_state.settings["tools_enabled"]["vector_store"] = st.checkbox(
        "Vector Search (for scraped content)",
        value=st.session_state.settings["tools_enabled"]["vector_store"]
    )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    st.session_state.settings["show_thinking"] = st.checkbox(
        "Show AI thinking process",
        value=st.session_state.settings["show_thinking"]
    )
    
    st.session_state.settings["max_iterations"] = st.slider(
        "Max Reasoning Steps",
        min_value=1,
        max_value=10,
        value=st.session_state.settings["max_iterations"]
    )
    
    st.session_state.settings["memory_messages"] = st.slider(
        "Memory Window (# of messages)",
        min_value=2,
        max_value=20,
        value=st.session_state.settings["memory_messages"]
    )
    
    # Option to clear chat history
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Conversation"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi, I'm an advanced search assistant that can search the web, scrape websites, access academic papers, and knowledge bases. I run locally using open-source models. How can I help you today?"}
            ]
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.success("Conversation cleared!")
    
    with col2:
        if st.button("Clear Vector Cache"):
            st.session_state.vector_stores = {}
            try:
                import shutil
                shutil.rmtree(".vector_cache", ignore_errors=True)
                os.makedirs(".vector_cache", exist_ok=True)
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")
            st.success("Vector cache cleared!")
    
    # Export chat history
    if st.session_state.conversation_history and st.button("Export Conversation"):
        chat_export = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation": st.session_state.conversation_history
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(chat_export, indent=2),
            file_name=f"conversation_export_{int(time.time())}.json",
            mime="application/json"
        )
    
    st.markdown("---")
    st.markdown("Built with [LangChain](https://www.langchain.com/), [Ollama](https://ollama.ai/) and [Streamlit](https://streamlit.io/)")
    st.markdown("Developed by [Subhash Gupta](https://subh24ai.github.io/)")

# Display vector stores (if any)
if st.session_state.vector_stores and st.session_state.settings["tools_enabled"]["vector_store"]:
    with st.expander("Scraped Websites (Vector Stores)"):
        for key, vs_data in st.session_state.vector_stores.items():
            st.write(f"📄 {vs_data['url']} - Scraped at {vs_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize tools
tools = []

# DuckDuckGo Search
if st.session_state.settings["tools_enabled"]["duckduckgo"]:
    search = DuckDuckGoSearchRun(name="Search")
    tools.append(search)

# arXiv
if st.session_state.settings["tools_enabled"]["arxiv"]:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1500)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    tools.append(arxiv)

# Wikipedia
if st.session_state.settings["tools_enabled"]["wikipedia"]:
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    tools.append(wiki)

# Web Scraper
if st.session_state.settings["tools_enabled"]["web_scraper"]:
    web_scraper = Tool(
        name="WebScraper",
        func=scrape_website,
        description="Useful for when you need to get the content from a specific webpage. Input should be a valid URL."
    )
    tools.append(web_scraper)

# Vector Search
if st.session_state.settings["tools_enabled"]["vector_store"]:
    vector_search_tool = Tool(
        name="VectorSearch",
        func=vector_search,
        description="Search through previously scraped websites for specific information. Input should be a search query."
    )
    tools.append(vector_search_tool)

# Function to check if Ollama is running and has the model
def check_ollama_status(server_url, model_name):
    import requests
    try:
        response = requests.get(f"{server_url}/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json()["models"]]
            if model_name in available_models:
                return True, f"✅ {model_name} is available"
            else:
                return False, f"⚠️ {model_name} not found. Run 'ollama pull {model_name}' in terminal."
        else:
            return False, "⚠️ Ollama server responded with an error."
    except Exception as e:
        return False, f"❌ Could not connect to Ollama server: {str(e)}"

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check Ollama status 
    model_name = st.session_state.settings["model"]
    server_url = st.session_state.settings["local_server"]
    model_ok, status_msg = check_ollama_status(server_url, model_name)
    
    with st.sidebar:
        model_status.markdown(f"**Model Status:** {status_msg}")
    
    if not model_ok:
        with st.chat_message("assistant"):
            st.error(f"Cannot proceed: {status_msg}")
            st.info("Make sure Ollama is running and the selected model is installed.")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {status_msg}. Please make sure Ollama is running and the selected model is installed."})
    else:
        # Create memory for conversation context
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            return_messages=True,
            k=st.session_state.settings["memory_messages"]
        )
        for message in st.session_state.chat_history[-st.session_state.settings["memory_messages"]:]:
            memory.chat_memory.add_message(message)
        
        # Initialize the LLM
        llm = ChatOllama(
            model=model_name,
            temperature=st.session_state.settings["temperature"],
            base_url=server_url,
            streaming=True,
            max_tokens=st.session_state.settings["max_tokens"]
        )
        
        # Configure the agent
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
            "system_message": SystemMessage(content=st.session_state.settings["system_prompt"])
        }
        
        # Initialize the agent
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=st.session_state.settings["max_iterations"]
        )
        
        # Process the conversation and display the assistant's response
        with st.chat_message("assistant"):
            response_container = st.container()
            
            # Configure callback handler based on user settings
            st_cb = StreamlitCallbackHandler(
                response_container,
                expand_new_thoughts=st.session_state.settings["show_thinking"]
            )
            
            # Run the agent with progress indicator and error handling
            try:
                with st.spinner("Searching and thinking..."):
                    response = search_agent.run(
                        input=prompt,
                        callbacks=[st_cb] if st.session_state.settings["show_thinking"] else None
                    )
                
                # Display the final response
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_history.append({"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%H:%M:%S")})
                
                # Update memory for future interactions
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.conversation_history.append({"role": "assistant", "content": error_msg, "timestamp": datetime.now().strftime("%H:%M:%S")})

# Information section
with st.expander("About this app"):
    st.markdown("""
    ### Advanced Research Assistant

    This application integrates multiple search tools with powerful open-source language models to provide comprehensive answers to your questions:
    
    - **Local AI**: Runs models locally using Ollama - no API keys needed
    - **Web Search**: Searches the internet for current information
    - **Wikipedia**: Retrieves factual information from Wikipedia
    - **Academic Papers**: Searches arXiv for scholarly articles and research papers
    - **Web Scraper**: Extracts detailed information from specific websites
    - **Vector Search**: Stores and searches previously scraped websites
    
    The assistant uses LangChain's agent framework to intelligently decide which tools to use based on your question, allowing for sophisticated reasoning and comprehensive answers.
    
    #### Getting Started:
    1. Make sure you have [Ollama](https://ollama.ai/) installed and running
    2. Pull your preferred model with `ollama pull llama3` (or other model name)
    3. Start asking questions!
    
    #### Tips for effective use:
    - Ask specific questions for more precise answers
    - For academic topics, ensure the arXiv tool is enabled
    - When you need information from a specific website, the assistant can use the web scraper
    - Adjust the temperature setting to control creativity vs. precision
    - Use the "Show AI thinking process" option to see how the assistant reaches its conclusions
    
    #### Limitations:
    - The assistant can only access information that is publicly available through its search tools
    - Academic paper searches are limited to what's available on arXiv
    - Web scraping may occasionally fail due to website restrictions or complex page structures
    - The quality of responses depends on the local model you've chosen
    """)