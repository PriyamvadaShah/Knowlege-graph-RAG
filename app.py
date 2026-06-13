import os
from pypdf import PdfReader

from langchain_neo4j import Neo4jGraph, LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from dotenv import load_dotenv

load_dotenv()

class KnowledgeGraphBackend:
    def __init__(self):
        """Initializes and verifies necessary environment configurations."""
        self.url = os.getenv("NEO4J_URL")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def _validate_config(self):
        if not self.url or not self.password or not self.openai_api_key:
            raise ValueError("Missing database credentials or OpenAI API key in environment variables.")

    def extract_text_from_pdf(self, file_path_or_buffer) -> str:
        """Extracts and cleans text content from an uploaded PDF file wrapper."""
        pdf_reader = PdfReader(file_path_or_buffer)
        extracted_text = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text)

    def ingest_text_to_graph(self, text: str):
        """Processes raw text through LLMGraphTransformer and stores it in Neo4j."""
        self._validate_config()
        
        graph = Neo4jGraph(
            url=self.url,
            username=self.username,
            password=self.password
        )
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.openai_api_key)
        transformer = LLMGraphTransformer(llm=llm)
        documents = [Document(page_content=text)]
        
        graph_documents = transformer.convert_to_graph_documents(documents)
        
        graph.add_graph_documents(graph_documents)

    def query_graph_rag(self, question: str) -> str:
        """Executes a hybrid structural GraphRAG search over the ingested knowledge base."""
        self._validate_config()
        
        driver = GraphDatabase.driver(
            self.url,
            auth=(self.username, self.password)
        )
        
        try:
            retriever = VectorCypherRetriever(
                driver=driver,
                retrieval_query="""
                MATCH (node)-[r]-(neighbor)
                RETURN node.name, type(r), neighbor.name LIMIT 15
                """
            )
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.openai_api_key)
            rag = GraphRAG(retriever=retriever, llm=llm)
            
            response = rag.search(question)
            return response.answer
            
        finally:
            driver.close()