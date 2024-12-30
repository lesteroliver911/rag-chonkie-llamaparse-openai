import os
import warnings
import streamlit as st
from llama_parse import LlamaParse
from chonkie import LateChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import List
import tempfile

# Environment and warning configurations
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# Initialize Chunker
chunker = LateChunker(
    embedding_model="all-MiniLM-L6-v2",
    mode="sentence",
    chunk_size=500,
    min_sentences_per_chunk=1,
    min_characters_per_sentence=12,
)

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
        )
        
    def parse_pdf(self, pdf_file) -> List[Document]:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Parse PDF using LlamaParse
            documents = self.parser.load_data(tmp_path)
            
            # Extract text content
            text_content = " ".join([doc.text for doc in documents])
            
            # Use LateChunker for chunking
            chunks = chunker(text_content)
            
            # Convert chunks to Langchain Documents
            langchain_docs = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.text,
                    metadata={
                        "token_count": chunk.token_count,
                        "sentence_count": len(chunk.sentences)
                    }
                )
                langchain_docs.append(doc)
            
            return langchain_docs
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        return FAISS.from_documents(documents, self.embeddings)

    def query_vector_store(self, vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
        return vector_store.similarity_search(query, k=k)

    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Use the following context to answer the question. If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content

def main():
    st.set_page_config(page_title="PDF Question Answering with RAG", page_icon="📚", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
            .small-font { font-size: 0.8em !important; }
            .metric-container { padding: 5px 0; }
            
            /* Chat container styling */
            .chat-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            
            /* Message bubbles */
            .question-bubble {
                background-color: #e6f3ff;
                color: #000000;
                padding: 15px;
                border-radius: 15px;
                margin: 10px 0;
                max-width: 80%;
                margin-left: auto;  /* Push to right */
            }
            
            .answer-bubble {
                background-color: #f0f2f6;
                color: #000000;
                padding: 15px;
                border-radius: 15px;
                margin: 10px 0;
                max-width: 80%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            
            /* Source chunk styling */
            .source-chunk {
                font-size: 0.9em;
                border-left: 3px solid #ccc;
                padding-left: 10px;
                margin: 5px 0;
                background-color: #ffffff;
                color: #000000;
            }

            /* Chat input container */
            .chat-input {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: calc(100% - 350px);  /* Adjust based on sidebar width */
                padding: 20px;
                background-color: white;
                border-top: 1px solid #eee;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.markdown("### 📑 Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                processor = DocumentProcessor()
                documents = processor.parse_pdf(uploaded_file)
                vector_store = processor.create_vector_store(documents)
                
                st.session_state.vector_store = vector_store
                st.session_state.processor = processor
                
                # Document statistics in sidebar
                st.markdown("<h4 style='font-size: 1.1em;'>📊 Document Statistics</h4>", unsafe_allow_html=True)
                total_tokens = sum(doc.metadata["token_count"] for doc in documents)
                total_sentences = sum(doc.metadata["sentence_count"] for doc in documents)
                
                st.markdown(f"""
                    <div class='small-font'>
                        <div class='metric-container'>
                            Chunks: {len(documents)}<br>
                            Total Tokens: {total_tokens}<br>
                            Avg. Chunk Size: {round(total_tokens/len(documents))}<br>
                            Total Sentences: {total_sentences}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Main content area
    if 'vector_store' not in st.session_state:
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h1>PDF Q&A using LlamaParse, Chonkie and OpenAI</h1>
                <p style='color: #666; font-size: 0.9em;'>Upload your PDF in the sidebar to begin</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Chat messages container
        chat_container = st.container()
        
        # Fixed input at bottom
        with st.container():
            st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
            query = st.text_input("Ask a question about the document", key="query_input")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display chat messages
        with chat_container:
            if query:
                with st.spinner("Generating answer..."):
                    relevant_docs = st.session_state.processor.query_vector_store(
                        st.session_state.vector_store, 
                        query
                    )
                    response = st.session_state.processor.generate_response(query, relevant_docs)
                    
                    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                    
                    # Question bubble
                    st.markdown(f"""
                        <div class='question-bubble'>
                            <strong>You:</strong><br>{query}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer bubble
                    st.markdown(f"""
                        <div class='answer-bubble'>
                            <strong>Assistant:</strong><br>{response}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Source chunks in an expander
                    with st.expander("📑 View source chunks"):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"""
                                <div class='source-chunk'>
                                    <strong>Source {i}</strong><br>
                                    {doc.page_content}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'")
        st.stop()
    
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        st.error("Please set your LlamaParse API key as an environment variable 'LLAMA_CLOUD_API_KEY'")
        st.stop()
    
    main()
