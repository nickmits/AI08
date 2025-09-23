import os
import nest_asyncio
import tiktoken
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Apply nested asyncio for Jupyter compatibility
nest_asyncio.apply()

# Define state
class State(TypedDict):
    question: str
    context: list[Document]
    response: str

# Document loading and processing
def load_and_process_documents():
    """Load and process documents from the data directory"""
    directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    ai_usage_knowledge_resources = directory_loader.load()
    
    # Text splitting with token-based approach
    def tiktoken_len(text):
        tokens = tiktoken.get_encoding("cl100k_base").encode(text)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    
    ai_usage_knowledge_chunks = text_splitter.split_documents(ai_usage_knowledge_resources)
    return ai_usage_knowledge_chunks

# Set up embeddings and vector store
def setup_vector_store():
    """Set up the vector store with embeddings"""
    # Embedding model
    embedding_model = OllamaEmbeddings(model="embeddinggemma:latest")
    embedding_dim = 768
    
    # Qdrant client and collection
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="ai_usage_knowledge_index",
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    
    # Vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="ai_usage_knowledge_index",
        embedding=embedding_model,
    )
    
    # Load documents and add to vector store
    ai_usage_knowledge_chunks = load_and_process_documents()
    vector_store.add_documents(documents=ai_usage_knowledge_chunks)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

# Initialize components
retriever = setup_vector_store()

# Chat prompt template
HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context response with "I don't know"
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("human", HUMAN_TEMPLATE)
])

# Ollama chat model
ollama_chat_model = ChatOllama(model="gpt-oss:20b", temperature=0.6)

# Define nodes
def retrieve(state: State) -> State:
    """Retrieve relevant documents based on the question"""
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}

def generate(state: State) -> State:
    """Generate response using the context and question"""
    generator_chain = chat_prompt | ollama_chat_model | StrOutputParser()
    response = generator_chain.invoke({"query": state["question"], "context": state["context"]})
    return {"response": response}

# Build graph
graph_builder = StateGraph(State)
graph_builder = graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

# Compile graph
graph = graph_builder.compile()

# Function to query the graph
def query_rag_system(question: str) -> str:
    """Query the RAG system with a question"""
    response = graph.invoke({"question": question})
    return response["response"]

if __name__ == "__main__":
    # Example usage
    print("RAG System initialized successfully!")
    print("\nExample queries:")
    
    # Test queries
    test_questions = [
        "What are the most common ways people use AI in their work?",
        "Do people use AI for their personal lives?",
        "Who is Batman?"  # This should return "I don't know"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = query_rag_system(question)
        print(f"A: {answer}")
