# 🔬 Advanced Research Assistant

An advanced AI research assistant that runs locally using open-source large language models. This application combines the power of modern LLMs with comprehensive search tools to provide intelligent answers with enhanced data retrieval and analysis capabilities - all while respecting your privacy by running everything locally.

![Advanced Research Assistant Screenshot](https://via.placeholder.com/800x450)

## 🌟 Features

### Local-First AI
- Runs entirely on your machine using [Ollama](https://ollama.ai/) - no API keys or cloud services required
- Support for multiple open-source models:
  - Llama 3 (Meta)
  - Phi-3 (Microsoft)
  - Mistral (Fast inference)
  - Gemma (Google)
  - Mixtral (Powerful reasoning)
  - Neural Chat (Intel)

### Comprehensive Research Tools
- **Web Search**: Real-time internet searches using DuckDuckGo
- **Academic Research**: Access to arXiv papers and scholarly content
- **Wikipedia**: Direct access to factual information
- **Web Scraper**: Extract detailed information from specific websites
- **Vector Store**: Cache and search previously scraped websites using embedding-based similarity search

### Advanced Capabilities
- **Intelligent Agent**: Uses LangChain's agent framework to intelligently decide which tools to use based on your question
- **Context-Aware Memory**: Maintains conversation history for coherent multi-turn discussions
- **Customizable System Prompt**: Fine-tune how the assistant thinks and behaves
- **Transparent Reasoning**: Option to view the AI's thinking process as it works on your query
- **Conversation Export**: Save your research sessions for future reference

## 🚀 Getting Started

### Prerequisites
- [Python](https://www.python.org/) (3.8 or higher)
- [Ollama](https://ollama.ai/) - for running local LLMs

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/advanced-research-assistant.git
   cd advanced-research-assistant
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Pull a model with Ollama
   ```bash
   ollama pull llama3:8b
   # Or another supported model
   ```

4. Run the application
   ```bash
   streamlit run appollama.py
   ```

## 💡 Usage Tips

- **Model Selection**: Choose the model that best fits your needs:
  - Llama 3 - Good all-rounder
  - Mistral - Faster responses
  - Mixtral - Best for complex reasoning
  - Phi-3 - Efficient knowledge model

- **Temperature Setting**: Adjust to control creativity (higher) vs. precision (lower)

- **Tool Selection**: Enable only the tools you need for more focused results

- **Web Scraping**: For detailed information from specific sources, ask the assistant to scrape a particular website

- **Vector Search**: After scraping websites, you can search through their content with natural language queries

## 🔧 Configuration

The application offers extensive customization options through the sidebar:
- Model selection and server URL
- Temperature and token settings
- Custom system prompts
- Tool enablement/disablement
- Memory window size
- Reasoning step limits

## 🧠 How It Works

1. **User Input**: You ask a question or make a request
2. **Agent Planning**: The LLM determines which tools are needed to answer
3. **Tool Execution**: The agent gathers information from selected sources
4. **Analysis**: The LLM processes and synthesizes the information
5. **Response**: A comprehensive answer is provided with citations

## 🔒 Privacy

This application prioritizes privacy:
- All LLM processing happens locally on your machine
- No data is sent to external AI providers
- Search queries use privacy-focused services when possible
- All scraped website data is stored locally

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the agent framework
- [Ollama](https://ollama.ai/) for local LLM hosting
- [Streamlit](https://streamlit.io/) for the web interface
- All the amazing open-source LLM developers

---

Developed by [Subhash Gupta](https://subh24ai.github.io/) #Research_Assistant
