# Advanced RAG + Raptor

An advanced Retrieval-Augmented Generation (RAG) system with Raptor integration for enhanced semantic search and document retrieval.

```mermaid
graph LR;

subgraph raptor;
ScrapedData --> RaptorPipLine
end
subgraph app
    %% direction ;
    InputQuery-->Contextualize-->QueryExpansion-->Retrieval-->ReRanking-->Output
end

    raptor --> app
```

## Overview

This project implements an advanced RAG system with a two-stage pipeline:

1. **Raptor Pipeline**: A hierarchical document processing system that takes scraped web content, converts it to markdown, and creates semantically clustered document embeddings with hierarchical summarization.

2. **Application**: A sophisticated query processing system that handles user input through contextualization, query expansion, retrieval, and re-ranking for accurate, contextually-relevant responses.

## Project Structure

```
Advanced RAG + Raptor
├── app/
│   ├── app.py               # Main Streamlit application
│   ├── embedding_pipline.py # Embedding pipeline for documents
│   └── utils.py             # Utility functions for query processing
└── raptor_pipeline/
    ├── convert_html2md.py   # HTML to Markdown converter
    ├── pipeline.py          # Document processing and embedding generation
    └── raptor.py            # Hierarchical clustering and summarization
```

## Features

### Raptor Pipeline

- **HTML to Markdown Conversion**: Converts HTML content to clean Markdown format
- **Document Processing**: Handles large document collections efficiently
- **Hierarchical Embedding**: Generates semantically rich embeddings
- **Clustering**: Groups similar documents using advanced clustering algorithms
- **Summarization**: Creates hierarchical summaries of document clusters

### Application Flow

- **Contextualization**: Understands query in context of conversation history
- **Query Expansion**: Generates multiple variations of the query to improve retrieval
- **Retrieval**: Fetches relevant documents from the vector store
- **Re-Ranking**: Prioritizes the most relevant documents using a cross-encoder model
- **Response Generation**: Creates concise, accurate responses based on retrieved information

## Technologies Used

- **LangChain**: Core framework for building the RAG pipeline
- **Google Vertex AI**: For embeddings (text-embedding-005) and LLM (gemini-2.0-flash)
- **PostgreSQL with pgvector**: Vector database for document storage and retrieval
- **UMAP & Gaussian Mixture Models**: For dimensionality reduction and clustering
- **Streamlit**: For the interactive user interface
- **Trafilatura**: For clean HTML content extraction

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Google Cloud Platform account with Vertex AI enabled

### Installation

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up environment variables:
```
PG_CONN="postgresql+psycopg://user:password@localhost:5432/database"
```

### Data Processing

1. Convert HTML documents to Markdown:
```
python raptor_pipeline/convert_html2md.py
```

2. Process documents through the Raptor pipeline:
```
python raptor_pipeline/pipeline.py
```

### Running the Application

Launch the Streamlit application:
```
streamlit run app/app.py
```

## How It Works

### Document Processing Flow

1. HTML documents are converted to Markdown format
2. Documents are split into manageable chunks
3. Chunks are embedded using Vertex AI embeddings
4. Embeddings are clustered using hierarchical clustering
5. Clusters are summarized to create hierarchical summaries
6. Final embeddings and metadata are stored in PostgreSQL with pgvector

### Query Processing Flow

1. User query is contextualized with conversation history
2. Query is expanded to create multiple variations
3. Expanded queries are used to retrieve relevant documents
4. Retrieved documents are re-ranked for relevance
5. Most relevant documents are used to generate a response
6. Response is presented to the user with source references

## Advanced Features

- **Recursive Clustering**: Multi-level document organization
- **Contextualized Queries**: Consideration of conversation history
- **Query Expansion**: Improved retrieval through query variations
- **Cross-Encoder Re-Ranking**: More accurate document relevance scoring
- **Source Attribution**: References to source documents in responses
