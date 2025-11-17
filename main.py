#!/usr/bin/env python3
"""
AmbedkarGPT - Q&A System using RAG Pipeline
Orchestrates the retrieval-augmented generation process using LangChain, ChromaDB, and Ollama
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

# Fix for ChromaDB Pydantic V1 compatibility with Python 3.14+
# Set environment variable before ChromaDB imports
if "CHROMA_SERVER_NOFILE" not in os.environ:
    os.environ["CHROMA_SERVER_NOFILE"] = "1024"

# LangChain imports - using compatibility imports for assignment requirements
try:
    # Try new LangChain 1.0+ imports first
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import ChatOllama
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    # Use langchain-huggingface for embeddings (recommended)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    USE_NEW_API = True
except ImportError:
    # Fallback to old LangChain 0.x imports (for assignment compatibility)
    from langchain.document_loaders import TextLoader
    from langchain.text_splitters import CharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import Ollama as ChatOllama
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    USE_NEW_API = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation System for Q&A based on document corpus
    """
    
    def __init__(
        self,
        document_path: str = "speech.txt",
        persist_dir: str = "./chroma_db",
        model_name: str = "mistral",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG system with configuration
        
        Args:
            document_path: Path to the document file
            persist_dir: Directory to persist ChromaDB
            model_name: Ollama model name to use
            embedding_model: HuggingFace embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.document_path = document_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.documents: List[Document] = []
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.llm = None
        
        logger.info(f"RAG System initialized with config: model={model_name}, chunk_size={chunk_size}")
    
    def load_documents(self) -> bool:
        """
        Load documents from the specified file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.document_path):
                logger.error(f"Document not found: {self.document_path}")
                return False
            
            logger.info(f"Loading document from: {self.document_path}")
            loader = TextLoader(self.document_path, encoding='utf-8')
            self.documents = loader.load()
            logger.info(f"Successfully loaded {len(self.documents)} document(s)")
            return True
        
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False
    
    def split_documents(self) -> List[Document]:
        """
        Split documents into manageable chunks
        
        Returns:
            List of split documents
        """
        try:
            logger.info(f"Splitting documents with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
            split_docs = splitter.split_documents(self.documents)
            logger.info(f"Successfully split into {len(split_docs)} chunks")
            return split_docs
        
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return []
    
    def create_embeddings(self) -> bool:
        """
        Create embeddings and store in vector database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            split_docs = self.split_documents()
            if not split_docs:
                return False
            
            logger.info(f"Creating embeddings using: {self.embedding_model}")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            
            logger.info(f"Storing embeddings in ChromaDB at: {self.persist_dir}")
            try:
                # Try with explicit client creation first
                import chromadb
                chroma_client = chromadb.PersistentClient(path=self.persist_dir)
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                    client=chroma_client,
                    collection_name="ambedkar_speeches"
                )
            except Exception as e1:
                # Fallback to standard initialization
                logger.warning(f"Client-based init failed, trying standard: {e1}")
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                    persist_directory=self.persist_dir,
                    collection_name="ambedkar_speeches"
                )
            
            logger.info("Vector store created successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def load_vector_store(self) -> bool:
        """
        Load existing vector store from disk
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.persist_dir):
                logger.info("Vector store not found, will create new one")
                return False
            
            logger.info(f"Loading vector store from: {self.persist_dir}")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            try:
                # Try with explicit client creation first
                import chromadb
                chroma_client = chromadb.PersistentClient(path=self.persist_dir)
                self.vector_store = Chroma(
                    client=chroma_client,
                    embedding_function=embeddings,
                    collection_name="ambedkar_speeches"
                )
            except Exception as e1:
                # Fallback to standard initialization
                logger.warning(f"Client-based load failed, trying standard: {e1}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=embeddings,
                    collection_name="ambedkar_speeches"
                )
            
            logger.info("Vector store loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def initialize_chain(self) -> bool:
        """
        Initialize the RAG chain with LLM and retriever
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.vector_store is None:
                logger.error("Vector store not initialized")
                return False
            
            # Initialize LLM
            logger.info(f"Initializing Ollama LLM with model: {self.model_name}")
            if USE_NEW_API:
                self.llm = ChatOllama(model=self.model_name, temperature=0.3)
            else:
                # Old API - ChatOllama is actually Ollama
                self.llm = ChatOllama(model=self.model_name, temperature=0.3)
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
            )
            
            # Create QA chain based on API version
            if USE_NEW_API:
                # New LangChain 1.0+ API using LCEL
                prompt = ChatPromptTemplate.from_template(
                    """Use the following pieces of context to answer the question. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    
                    Context: {context}
                    
                    Question: {question}
                    
                    Answer:"""
                )
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                self.qa_chain = (
                    {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
            else:
                # Old LangChain 0.x API using RetrievalQA
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True,
                    verbose=False
                )
            
            logger.info("QA chain initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            return False
    
    def setup(self) -> bool:
        """
        Complete setup: load/create vector store and initialize chain
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Try loading existing vector store
        if not self.load_vector_store():
            # Create new vector store if it doesn't exist
            if not self.load_documents():
                return False
            if not self.create_embeddings():
                return False
        
        # Initialize the QA chain
        return self.initialize_chain()
    
    def answer_question(self, question: str) -> dict:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            logger.error("QA chain not initialized. Run setup() first.")
            return {"error": "System not initialized"}
        
        try:
            logger.info(f"Processing question: {question}")
            
            if USE_NEW_API:
                # New API - chain returns string directly
                answer = self.qa_chain.invoke(question)
                # Get source documents separately
                source_docs = self.retriever.invoke(question)
            else:
                # Old API - RetrievalQA returns dict with "result" and "source_documents"
                result = self.qa_chain({"query": question})
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
            
            logger.info("Question answered successfully")
            return {
                "question": question,
                "answer": answer,
                "sources": source_docs
            }
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {"error": str(e)}
    
    def batch_questions(self, questions: List[str]) -> List[dict]:
        """
        Answer multiple questions in batch
        
        Args:
            questions: List of questions
            
        Returns:
            List of answers
        """
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            results.append(self.answer_question(question))
        
        return results


def main():
    """Main entry point for the RAG system"""
    
    logger.info("=" * 50)
    logger.info("AmbedkarGPT - RAG Q&A System Starting")
    logger.info("=" * 50)
    
    # Initialize system
    rag = RAGSystem(
        document_path="speech.txt",
        persist_dir="./chroma_db",
        model_name="mistral"
    )
    
    # Setup
    if not rag.setup():
        logger.error("Failed to setup RAG system")
        sys.exit(1)
    
    logger.info("RAG system ready. Entering interactive mode.")
    logger.info("Type 'exit' or 'quit' to exit")
    logger.info("-" * 50)
    
    # Interactive loop
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                logger.info("Exiting...")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            result = rag.answer_question(question)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nAnswer: {result['answer']}")
                if result.get('sources'):
                    print("\nSource Documents:")
                    for i, doc in enumerate(result['sources'], 1):
                        print(f"  [{i}] {doc.page_content[:100]}...")
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print(f"Error: {str(e)}")
    
    logger.info("=" * 50)
    logger.info("AmbedkarGPT Session Ended")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()