import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

# Core libraries
import pandas as pd
import numpy as np
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions

# LangGraph and LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
import operator

# LLM and embeddings
from sentence_transformers import SentenceTransformer
from groq import Groq

# Web search
import duckduckgo_search as ddgs
from duckduckgo_search import DDGS

# Reranking
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    content: str
    source: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    query: str
    vector_results: List[SearchResult]
    web_results: List[SearchResult]
    final_response: str
    sources: List[str]
    retrieved_docs: List[str]
    needs_web_search: bool
    error: Optional[str]

class AgenticRAGLawAssistant:
    """
    Agentic RAG Law Assistant with vector database and web search capabilities
    """
    
    def __init__(self, 
                 groq_api_key: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 vector_db_path: str = "./law_vectordb",
                 top_k_vector: int = 5,
                 top_k_web: int = 3,
                 relevance_threshold: float = 0.6,
                 groq_model: str = "llama3-8b-8192"):
        """
        Initialize the Law Assistant
        
        Args:
            groq_api_key: Groq API key
            embedding_model: SentenceTransformer model for embeddings
            reranking_model: Cross-encoder model for reranking
            vector_db_path: Path to store vector database
            top_k_vector: Number of top results from vector search
            top_k_web: Number of top results from web search
            relevance_threshold: Threshold for determining if vector results are sufficient
            groq_model: Groq model to use (llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it)
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranking_model)
        self.vector_db_path = vector_db_path
        self.top_k_vector = top_k_vector
        self.top_k_web = top_k_web
        self.relevance_threshold = relevance_threshold
        
        # Initialize vector database
        self.vector_db = None
        self.collection = None
        self._initialize_vector_db()
        
        # Initialize LangGraph workflow
        self.workflow = None
        self._build_workflow()
        
        logger.info("AgenticRAGLawAssistant initialized successfully")
    
    def _initialize_vector_db(self):
        """Initialize and populate vector database with Indian law dataset"""
        try:
            # Initialize ChromaDB
            self.vector_db = chromadb.PersistentClient(path=self.vector_db_path)
            
            # Create or get collection
            try:
                self.collection = self.vector_db.get_collection("indian_law")
                logger.info("Loaded existing vector database")
            except:
                logger.info("Creating new vector database from dataset...")
                self._create_vector_db_from_dataset()
                
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
    
    def _create_vector_db_from_dataset(self):
        """Create vector database from HuggingFace dataset"""
        try:
            # Load dataset
            logger.info("Loading Indian law dataset...")
            dataset = load_dataset("viber1/indian-law-dataset")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset['train'])
            
            # Create collection
            self.collection = self.vector_db.create_collection(
                name="indian_law",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                # Prepare documents for indexing
                documents = batch['instruction'].tolist()
                responses = batch['response'].tolist()
                
                # Create IDs
                ids = [f"doc_{i+j}" for j in range(len(documents))]
                
                # Create metadata
                metadatas = [
                    {
                        "response": response,
                        "instruction": instruction,
                        "doc_id": doc_id
                    }
                    for doc_id, instruction, response in zip(ids, documents, responses)
                ]
                
                # Add to collection
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            logger.info("Vector database created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
            raise
    
    def _vector_search(self, query: str) -> List[SearchResult]:
        """Perform vector search and rerank results"""
        try:
            # Search vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=self.top_k_vector * 2  # Get more results for reranking
            )
            
            # Prepare results for reranking
            search_results = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                search_results.append(SearchResult(
                    content=metadata['response'],
                    source="Vector Database",
                    metadata=metadata
                ))
            
            # Rerank results
            if search_results:
                pairs = [(query, result.content) for result in search_results]
                scores = self.reranker.predict(pairs)
                
                # Update relevance scores and sort
                for result, score in zip(search_results, scores):
                    result.relevance_score = float(score)
                
                search_results.sort(key=lambda x: x.relevance_score, reverse=True)
                search_results = search_results[:self.top_k_vector]
            
            logger.info(f"Vector search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def _web_search(self, query: str) -> List[SearchResult]:
        """Perform web search using DuckDuckGo"""
        try:
            # Add legal context to query
            legal_query = f"Indian law legal {query}"
            
            # Perform web search
            with DDGS() as ddgs:
                results = list(ddgs.text(legal_query, max_results=self.top_k_web))
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    content=result.get('body', ''),
                    source=result.get('href', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'href': result.get('href', '')
                    }
                ))
            
            logger.info(f"Web search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def _build_workflow(self):
        """Build LangGraph workflow for agentic behavior"""
        
        def vector_search_node(state: AgentState) -> AgentState:
            """Node for vector search"""
            logger.info("Executing vector search...")
            vector_results = self._vector_search(state["query"])
            
            # Check if results are sufficient
            avg_relevance = np.mean([r.relevance_score for r in vector_results]) if vector_results else 0
            needs_web_search = avg_relevance < self.relevance_threshold or len(vector_results) < 2
            
            return {
                **state,
                "vector_results": vector_results,
                "needs_web_search": needs_web_search
            }
        
        def web_search_node(state: AgentState) -> AgentState:
            """Node for web search"""
            logger.info("Executing web search...")
            web_results = self._web_search(state["query"])
            
            return {
                **state,
                "web_results": web_results
            }
        
        def llm_generation_node(state: AgentState) -> AgentState:
            """Node for LLM generation"""
            logger.info("Generating final response...")
            
            # Combine all results
            all_results = state.get("vector_results", []) + state.get("web_results", [])
            
            # Prepare context
            context = ""
            sources = []
            retrieved_docs = []
            
            for i, result in enumerate(all_results):
                context += f"Source {i+1}: {result.content}\n\n"
                sources.append(result.source)
                retrieved_docs.append(result.content)
            
            # Generate response using Groq
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert Indian legal assistant. Provide accurate, helpful responses about Indian law based on the provided context. 
                            Always cite your sources and be precise in your legal explanations. If you're uncertain about something, say so.
                            Format your response in a clear, structured manner. Use proper legal terminology and provide practical guidance where applicable."""
                        },
                        {
                            "role": "user",
                            "content": f"""Question: {state['query']}
                            
                            Context from legal sources:
                            {context}
                            
                            Please provide a comprehensive answer based on the context above. Structure your response with:
                            1. Direct answer to the question
                            2. Legal basis/provisions
                            3. Practical implications
                            4. Any relevant case law or examples (if available in context)"""
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    top_p=1,
                    stream=False
                )
                
                final_response = response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"Error generating Groq response: {str(e)}")
                final_response = "I apologize, but I encountered an error while generating the response. Please try again."
                
            return {
                **state,
                "final_response": final_response,
                "sources": sources,
                "retrieved_docs": retrieved_docs
            }
        
        def should_web_search(state: AgentState) -> str:
            """Conditional edge to determine if web search is needed"""
            if state.get("needs_web_search", False):
                return "web_search"
            else:
                return "llm_generation"
        
        # Build workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("vector_search", vector_search_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("llm_generation", llm_generation_node)
        
        # Add edges
        workflow.set_entry_point("vector_search")
        workflow.add_conditional_edges(
            "vector_search",
            should_web_search,
            {
                "web_search": "web_search",
                "llm_generation": "llm_generation"
            }
        )
        workflow.add_edge("web_search", "llm_generation")
        workflow.add_edge("llm_generation", END)
        
        # Compile workflow
        memory = MemorySaver()
        self.workflow = workflow.compile(checkpointer=memory)
        
        logger.info("Workflow built successfully")
    
    def ask_question(self, query: str) -> Dict[str, Any]:
        """
        Main function to ask a question to the law assistant
        
        Args:
            query: The legal question to ask
            
        Returns:
            Dictionary containing response, sources, and retrieved documents
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Initialize state
            initial_state = {
                "query": query,
                "vector_results": [],
                "web_results": [],
                "final_response": "",
                "sources": [],
                "retrieved_docs": [],
                "needs_web_search": False,
                "error": None
            }
            
            # Run workflow
            config = {"configurable": {"thread_id": "law_assistant_thread"}}
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Prepare response
            response = {
                "response": final_state.get("final_response", ""),
                "sources": final_state.get("sources", []),
                "retrieved_docs": final_state.get("retrieved_docs", []),
                "vector_results": final_state.get("vector_results", []),
                "web_results": final_state.get("web_results", []),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "retrieved_docs": [],
                "vector_results": [],
                "web_results": [],
                "timestamp": datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Initialize the assistant
    # Note: You'll need to set your Groq API key
    assistant = AgenticRAGLawAssistant(
        groq_api_key="your-groq-api-key-here",
        groq_model="llama3-8b-8192"  # or "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
    )
    
    # Ask a question
    result = assistant.ask_question("What are the rights of consumers under Indian law?")
    
    print("Response:", result["response"])
    print("Sources:", result["sources"])
    print("Retrieved Documents:", len(result["retrieved_docs"]))