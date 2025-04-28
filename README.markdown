# GraphRAG vs RAG

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-4.44.0-green)
![Neo4j](https://img.shields.io/badge/Neo4j-5.24.0-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-1.12.0-red)

## Overview

GraphRAG vs RAG is a platform for processing and querying PDF documents using a hybrid approach that combines vector-based (RAG) and graph-based (GraphRAG) methods. It leverages machine learning and database technologies to analyze documents and extract relationships, comparing vector, graph, and hybrid query results.

## Technical Approach

The system integrates vector and graph databases for document processing and querying.

### System Architecture

GraphRAG vs RAG includes:

- **VectorDBProcessor**: Handles text chunking, embeddings, and vector storage for RAG queries.
- **GraphDBProcessor**: Extracts entities and relationships for GraphRAG queries.
- **HybridProcessor**: Combines vector and graph results for unified responses.
- **Gradio Interface**: Provides a web interface for document uploads, queries, and result comparison.

### Vector-Based Processing (RAG)

The RAG approach focuses on semantic text analysis.

- **Text Extraction**: Extracts text from PDFs using `PyMuPDF`.
- **Chunking and Embeddings**: Splits text into chunks and generates embeddings with OpenAI’s `text-embedding-3-small` model.
- **Storage and Querying**: Stores embeddings in [Qdrant](https://qdrant.tech/) and retrieves relevant chunks for query answers via OpenAI’s `gpt-4o-mini`.

### Graph-Based Processing (GraphRAG)

The GraphRAG approach extracts structured relationships.

- **Entity Extraction**: Identifies entities (e.g., Person, Organization) and relationships (e.g., WORKS_FOR) using OpenAI’s `gpt-4o-mini`.
- **Storage and Querying**: Stores data in [Neo4j](https://neo4j.com/) and generates Cypher queries for relationship retrieval.

### Hybrid Processing

Combines RAG and GraphRAG for comprehensive query responses, merging semantic and structured data.

### User Interface

A [Gradio](https://www.gradio.app/)-based web interface supports:

- PDF uploads and management.
- Structured query input.
- Comparison of RAG, GraphRAG, and hybrid results.

### Technologies Used

- **Python Libraries**: `gradio`, `pymupdf`, `openai`, `qdrant-client`, `neo4j`, `scikit-learn`, `python-dotenv`.
- **Databases**: Qdrant (vector), Neo4j (graph).
- **APIs**: OpenAI (`text-embedding-3-small`, `gpt-4o-mini`).
- **Environment**: Configured via `.env`.

### Error Handling and Logging

- Uses Python’s `logging` for tracking events and errors.
- Implements try-catch blocks for robust error handling.
- Displays user-friendly error messages in the interface.

## Conclusion

GraphRAG vs RAG integrates RAG and GraphRAG techniques, powered by OpenAI models, to offer a versatile document analysis platform. Its Gradio interface ensures transparency, making it suitable for knowledge extraction applications.