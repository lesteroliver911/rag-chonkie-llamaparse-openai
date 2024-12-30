# RAG Development Accelerator with Chonkie

A practical implementation showcasing the power of late chunking strategy using Chonkie, combined with LlamaParse and LangChain. This project serves as a learning tool and development accelerator for building efficient RAG (Retrieval-Augmented Generation) applications.

![Application Demo](chonkie-latechunking-llamaparse-openai-demo.gif)

## Purpose

This project demonstrates how to build a lightweight, efficient RAG application using:

- **Chonkie**: A no-nonsense chunking library that implements late chunking strategy for optimal performance
- **LlamaParse**: Advanced document parsing capabilities
- **LangChain**: Flexible framework for LLM application development

### Why Late Chunking?

Late chunking, as implemented by Chonkie, provides several advantages:
- Faster processing
- Lower memory usage
- Better context preservation
- Dynamic chunk size adaptation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lesteroliver911/rag-chonkie-llamaparse-openai
cd pdf-qa-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install streamlit llama-parse "chonkie[st]" langchain-openai faiss-cpu python-dotenv
```

## Configuration

1. Create a `.env` file in the project root:
```bash
touch .env
```

2. Add your API keys to the `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
LLAMA_CLOUD_API_KEY=your-llamaparse-api-key
```

3. Create necessary project directories:
```bash
mkdir tmp
mkdir data
```

## File Structure
```
pdf-qa-rag/
├── app.py
├── .env
├── requirements.txt
├── README.md
├── tmp/
└── data/
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. Upload a PDF document using the file uploader

4. Wait for the document to be processed

5. Enter your questions in the text input field

## 💡 Key Implementation Details

### Chonkie Integration
```python
from chonkie import TokenChunker
from tokenizers import Tokenizer

# Initialize tokenizer and chunker
tokenizer = Tokenizer.from_pretrained("gpt2")
chunker = TokenChunker(tokenizer)

# Process text with late chunking
chunks = chunker("Your text here")
for chunk in chunks:
    print(f"Chunk size: {chunk.token_count}")
```

### LlamaParse PDF Processing
```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key=llama_api_key,
    result_type="markdown",
    verbose=True
)

# Async PDF parsing
documents = await parser.aload_data("your_file.pdf")
```

### Context-Aware Response Generation
```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides clear and concise responses."),
    ("human", "{context}\n\nQuestion: {input}")
])

response = chain.invoke({
    "input": combined_input,
    "context": context
})
```

## 🎓 Learning Points

1. **Late Chunking Strategy**
   - Chunks are processed only when needed
   - Maintains document context integrity
   - Reduces processing overhead

2. **PDF Processing Pipeline**
   ```mermaid
   graph LR
   A[PDF Upload] --> B[LlamaParse]
   B --> C[Text Extraction]
   C --> D[Chonkie Chunking]
   D --> E[Context Storage]
   ```

3. **Token Management**
   - Real-time token counting
   - Usage optimization
   - Cost monitoring

## 🔧 Customization Options

### Chunk Size Adjustment
```python
# Modify chunk settings
chunker = TokenChunker(
    tokenizer,
    chunk_size=500,  # Default is 1000
    overlap=50       # Default is 200
)
```

### Model Selection
```python
# Change OpenAI model
llm = ChatOpenAI(
    model_name="gpt-4",  # Default is gpt-3.5-turbo
    temperature=0.7
)
```

## Resources

- [Chonkie Documentation](https://github.com/chonkie-ai/chonkie)
- [LlamaParse Guide](https://cloud.llamaindex.ai/)
- [LangChain Framework](https://python.langchain.com/docs/get_started/introduction)

## License

MIT License - feel free to use this in your projects!

---
Built with ❤️ for developers who want to accelerate their RAG implementations.
