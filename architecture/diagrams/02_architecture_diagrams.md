# Architecture Diagrams

## Overview

This document provides comprehensive architecture diagrams for the Advanced Retrieval project (aie8-s09-adv-retrieval). The project focuses on exploring and comparing multiple retrieval strategies for RAG (Retrieval-Augmented Generation) systems using LangChain.

The main codebase consists of:
- **config.py**: Configuration management for paths and settings
- **adv-retrieval.py**: Core retrieval implementation (Jupyter notebook in Python format)
- **notebooks/session09-adv-retrieval.ipynb**: Interactive notebook demonstrating retrieval strategies

The project implements 7 distinct retrieval strategies with a common RAG chain pattern, enabling side-by-side comparison of retrieval performance.

---

## System Architecture

The system follows a layered architecture pattern, organizing functionality from configuration and data loading at the base, through retrieval strategies in the middle, to RAG chains at the top.

```mermaid
graph TB
    subgraph "Configuration Layer"
        CONFIG[config.py<br/>Project Paths & Settings]
    end

    subgraph "Data Layer"
        CSV[CSV Data Loader<br/>Projects_with_Domains.csv]
        DOCS[Document Objects<br/>synthetic_usecase_data]
    end

    subgraph "Embedding Layer"
        EMBED[OpenAI Embeddings<br/>text-embedding-3-small]
    end

    subgraph "Vector Store Layer"
        VS1[Naive VectorStore<br/>Qdrant - Synthetic_Usecases]
        VS2[Semantic VectorStore<br/>Qdrant - Semantic_Chunks]
        VS3[Parent VectorStore<br/>Qdrant - full_documents]
        VS4[InMemoryStore<br/>Parent Document Storage]
    end

    subgraph "Retrieval Strategy Layer"
        R1[Naive Retriever<br/>k=10]
        R2[BM25 Retriever<br/>Bag-of-Words]
        R3[Multi-Query Retriever<br/>LLM-generated queries]
        R4[Parent Document Retriever<br/>Small-to-Big strategy]
        R5[Compression Retriever<br/>Cohere Rerank]
        R6[Ensemble Retriever<br/>Reciprocal Rank Fusion]
        R7[Semantic Retriever<br/>Percentile chunking]
    end

    subgraph "RAG Chain Layer"
        CHAIN1[Naive Chain]
        CHAIN2[BM25 Chain]
        CHAIN3[Multi-Query Chain]
        CHAIN4[Parent Document Chain]
        CHAIN5[Compression Chain]
        CHAIN6[Ensemble Chain]
        CHAIN7[Semantic Chain]
    end

    subgraph "Generation Layer"
        LLM[ChatOpenAI<br/>gpt-4.1-nano]
        PROMPT[RAG Prompt Template]
    end

    subgraph "Output Layer"
        RESPONSE[Generated Response]
    end

    CONFIG --> CSV
    CSV --> DOCS
    DOCS --> EMBED
    EMBED --> VS1
    EMBED --> VS2
    EMBED --> VS3
    DOCS --> VS4

    VS1 --> R1
    DOCS --> R2
    VS1 --> R3
    VS3 --> R4
    VS4 --> R4
    VS1 --> R5
    VS2 --> R7
    R1 --> R6
    R2 --> R6
    R3 --> R6
    R4 --> R6
    R5 --> R6

    R1 --> CHAIN1
    R2 --> CHAIN2
    R3 --> CHAIN3
    R4 --> CHAIN4
    R5 --> CHAIN5
    R6 --> CHAIN6
    R7 --> CHAIN7

    CHAIN1 --> PROMPT
    CHAIN2 --> PROMPT
    CHAIN3 --> PROMPT
    CHAIN4 --> PROMPT
    CHAIN5 --> PROMPT
    CHAIN6 --> PROMPT
    CHAIN7 --> PROMPT

    PROMPT --> LLM
    LLM --> RESPONSE
```

### Key Insights

1. **Layered Separation**: Clear separation between data loading, embedding, retrieval, and generation layers enables modularity and testability
2. **Multiple Vector Stores**: The architecture uses 3 distinct Qdrant vector stores optimized for different retrieval strategies (naive, semantic, parent-document)
3. **Shared Components**: All retrieval chains share the same LLM (gpt-4.1-nano) and prompt template, isolating the retrieval strategy as the only variable
4. **Ensemble Pattern**: The Ensemble Retriever acts as a meta-retriever, combining outputs from 5 different retrieval strategies using Reciprocal Rank Fusion

---

## Component Relationships

This diagram shows how the core components interact and their dependencies. The relationships reveal a hub-and-spoke pattern with the document corpus at the center.

```mermaid
graph LR
    subgraph External Services
        OPENAI[OpenAI API<br/>Embeddings & LLM]
        COHERE[Cohere API<br/>Rerank Model]
    end

    subgraph Core Components
        DATA[CSV Loader<br/>Document Source]
        DOCS[Document Corpus<br/>synthetic_usecase_data]
        EMB[Embedding Model<br/>text-embedding-3-small]
        LLM[Chat Model<br/>gpt-4.1-nano]
    end

    subgraph Retriever Components
        NAIVE[Naive Retriever]
        BM25[BM25 Retriever]
        MULTI[Multi-Query Retriever]
        PARENT[Parent Doc Retriever]
        COMPRESS[Compression Retriever]
        ENSEMBLE[Ensemble Retriever]
        SEMANTIC[Semantic Retriever]
    end

    subgraph Storage Components
        QDRANT1[Qdrant VectorStore 1]
        QDRANT2[Qdrant VectorStore 2]
        QDRANT3[Qdrant VectorStore 3]
        MEMSTORE[InMemoryStore]
    end

    subgraph Chain Components
        CHAINS[7x RAG Chains]
        PROMPT[Prompt Template]
    end

    OPENAI --> EMB
    OPENAI --> LLM
    COHERE --> COMPRESS

    DATA --> DOCS
    DOCS --> EMB
    DOCS --> BM25

    EMB --> QDRANT1
    EMB --> QDRANT2
    EMB --> QDRANT3
    DOCS --> MEMSTORE

    QDRANT1 --> NAIVE
    QDRANT2 --> SEMANTIC
    QDRANT3 --> PARENT
    MEMSTORE --> PARENT
    DOCS --> BM25
    NAIVE --> MULTI
    LLM --> MULTI
    NAIVE --> COMPRESS

    NAIVE --> ENSEMBLE
    BM25 --> ENSEMBLE
    MULTI --> ENSEMBLE
    PARENT --> ENSEMBLE
    COMPRESS --> ENSEMBLE

    NAIVE --> CHAINS
    BM25 --> CHAINS
    MULTI --> CHAINS
    PARENT --> CHAINS
    COMPRESS --> CHAINS
    ENSEMBLE --> CHAINS
    SEMANTIC --> CHAINS

    CHAINS --> PROMPT
    PROMPT --> LLM
```

### Key Insights

1. **External Dependencies**: The system depends on two external APIs (OpenAI for embeddings/LLM, Cohere for reranking)
2. **Document Hub Pattern**: The document corpus serves as a central hub, feeding into multiple retrieval strategies
3. **Retriever Composition**: Some retrievers are built on top of others (Compression uses Naive, Multi-Query uses Naive, Ensemble uses 5 retrievers)
4. **Storage Diversity**: Different retrieval strategies require different storage mechanisms (3 vector stores + 1 in-memory store)
5. **Chain Uniformity**: All 7 RAG chains use identical prompt templates and LLM configuration, ensuring fair comparison

---

## Class Hierarchies

The project primarily leverages existing LangChain classes rather than defining custom class hierarchies. Below is the class structure showing the imported LangChain components and their relationships.

```mermaid
classDiagram
    class BaseRetriever {
        <<interface>>
        +get_relevant_documents()
    }

    class VectorStoreRetriever {
        +vectorstore
        +search_kwargs
    }

    class BM25Retriever {
        +from_documents()
        +k: int
    }

    class MultiQueryRetriever {
        +from_llm()
        +retriever
        +llm
    }

    class ParentDocumentRetriever {
        +vectorstore
        +docstore
        +child_splitter
        +add_documents()
    }

    class ContextualCompressionRetriever {
        +base_compressor
        +base_retriever
    }

    class EnsembleRetriever {
        +retrievers: list
        +weights: list
    }

    class BaseVectorStore {
        <<interface>>
        +from_documents()
        +as_retriever()
    }

    class Qdrant {
        +collection_name
        +location
        +embeddings
    }

    class QdrantVectorStore {
        +client
        +collection_name
        +embedding
    }

    class InMemoryStore {
        +mset()
        +mget()
    }

    class OpenAIEmbeddings {
        +model: str
        +embed_documents()
        +embed_query()
    }

    class ChatOpenAI {
        +model: str
        +invoke()
    }

    class CohereRerank {
        +model: str
        +compress_documents()
    }

    class SemanticChunker {
        +embeddings
        +breakpoint_threshold_type
        +split_documents()
    }

    BaseRetriever <|-- VectorStoreRetriever
    BaseRetriever <|-- BM25Retriever
    BaseRetriever <|-- MultiQueryRetriever
    BaseRetriever <|-- ParentDocumentRetriever
    BaseRetriever <|-- ContextualCompressionRetriever
    BaseRetriever <|-- EnsembleRetriever

    BaseVectorStore <|-- Qdrant
    BaseVectorStore <|-- QdrantVectorStore

    VectorStoreRetriever --> Qdrant : uses
    ParentDocumentRetriever --> QdrantVectorStore : uses
    ParentDocumentRetriever --> InMemoryStore : uses
    ContextualCompressionRetriever --> CohereRerank : uses
    MultiQueryRetriever --> ChatOpenAI : uses
    EnsembleRetriever --> BaseRetriever : aggregates

    Qdrant --> OpenAIEmbeddings : requires
    QdrantVectorStore --> OpenAIEmbeddings : requires
    SemanticChunker --> OpenAIEmbeddings : requires
```

### Key Insights

1. **No Custom Classes**: The project uses composition of existing LangChain classes rather than inheritance or custom implementations
2. **Retriever Polymorphism**: All retrievers implement the BaseRetriever interface, enabling consistent usage across RAG chains
3. **Strategy Pattern**: Different retriever implementations represent different retrieval strategies, following the Strategy design pattern
4. **Dependency Injection**: Retrievers receive their dependencies (embeddings, LLMs, stores) through constructor injection
5. **Framework-Driven**: The architecture relies heavily on LangChain's framework classes, minimizing custom code

---

## Module Dependencies

This diagram illustrates the dependency relationships between project modules and external packages.

```mermaid
graph TD
    subgraph "Project Modules"
        CONFIG[config.py<br/>Configuration]
        MAIN[adv-retrieval.py<br/>Main Implementation]
        NOTEBOOK[session09-adv-retrieval.ipynb<br/>Interactive Demo]
    end

    subgraph "LangChain Core"
        LC_CORE[langchain_core<br/>prompts, runnables, parsers]
    end

    subgraph "LangChain Community"
        LC_COMMUNITY[langchain_community<br/>document_loaders, retrievers, vectorstores]
    end

    subgraph "LangChain Integrations"
        LC_OPENAI[langchain_openai<br/>ChatOpenAI, OpenAIEmbeddings]
        LC_COHERE[langchain_cohere<br/>CohereRerank]
        LC_QDRANT[langchain_qdrant<br/>QdrantVectorStore]
        LC_EXPERIMENTAL[langchain_experimental<br/>SemanticChunker]
    end

    subgraph "LangChain Base"
        LC_BASE[langchain<br/>retrievers, storage]
    end

    subgraph "Vector Database"
        QDRANT[qdrant_client<br/>QdrantClient, models]
    end

    subgraph "External APIs"
        OPENAI_API[OpenAI API<br/>GPT-4.1-nano, Embeddings]
        COHERE_API[Cohere API<br/>Rerank v3.5]
    end

    subgraph "Python Standard Library"
        STD[os, getpass, datetime,<br/>operator, uuid, pathlib]
    end

    CONFIG --> STD
    MAIN --> STD
    MAIN --> LC_CORE
    MAIN --> LC_COMMUNITY
    MAIN --> LC_OPENAI
    MAIN --> LC_COHERE
    MAIN --> LC_QDRANT
    MAIN --> LC_EXPERIMENTAL
    MAIN --> LC_BASE
    MAIN --> QDRANT
    NOTEBOOK --> MAIN

    LC_OPENAI --> OPENAI_API
    LC_COHERE --> COHERE_API
    LC_QDRANT --> QDRANT
    LC_COMMUNITY --> QDRANT

    style CONFIG fill:#e1f5e1
    style MAIN fill:#e1f5e1
    style NOTEBOOK fill:#e1f5e1
```

### Key Insights

1. **Minimal Custom Code**: Only 2 Python modules (config.py and adv-retrieval.py) with config having zero external dependencies
2. **Heavy Framework Reliance**: The project depends on 6 different LangChain packages, showing deep integration with the framework
3. **External API Coupling**: Direct dependencies on OpenAI and Cohere APIs through LangChain wrappers
4. **Vector Store Coupling**: Tight coupling to Qdrant as the vector database (used through both langchain_community and langchain_qdrant)
5. **Low Coupling in Config**: config.py only uses Python standard library (pathlib), making it highly portable
6. **Notebook as Consumer**: The Jupyter notebook depends on the main module, suggesting proper code organization

---

## Data Flow

This diagram shows how data flows through the system from initial CSV loading through to final response generation.

```mermaid
flowchart TD
    START([User Query]) --> QUERY{Which Chain?}

    subgraph Data Preparation
        CSV[CSV File<br/>Projects_with_Domains.csv] --> LOADER[CSVLoader]
        LOADER --> |metadata extraction| DOCS[Document Objects<br/>page_content + metadata]
    end

    subgraph Embedding & Storage
        DOCS --> |embed| EMB[OpenAI Embeddings<br/>text-embedding-3-small]
        EMB --> |1536-dim vectors| VS[Vector Stores<br/>Qdrant Collections]
    end

    QUERY --> |1| NAIVE_FLOW[Naive Retrieval]
    QUERY --> |2| BM25_FLOW[BM25 Retrieval]
    QUERY --> |3| MULTI_FLOW[Multi-Query Retrieval]
    QUERY --> |4| PARENT_FLOW[Parent Doc Retrieval]
    QUERY --> |5| COMPRESS_FLOW[Compression Retrieval]
    QUERY --> |6| ENSEMBLE_FLOW[Ensemble Retrieval]
    QUERY --> |7| SEMANTIC_FLOW[Semantic Retrieval]

    subgraph Naive Flow
        NAIVE_FLOW --> |embed query| NV1[Query Vector]
        NV1 --> |cosine similarity| NV2[Vector Store Search]
        VS --> NV2
        NV2 --> |top k=10| NV3[Retrieved Docs]
    end

    subgraph BM25 Flow
        BM25_FLOW --> |tokenize| BM1[Query Tokens]
        BM1 --> |bag-of-words| BM2[BM25 Scoring]
        DOCS --> BM2
        BM2 --> |top k| BM3[Retrieved Docs]
    end

    subgraph Multi-Query Flow
        MULTI_FLOW --> |LLM expansion| MQ1[Generate N Queries]
        MQ1 --> |for each query| MQ2[Retrieve Documents]
        VS --> MQ2
        MQ2 --> |deduplicate| MQ3[Unique Retrieved Docs]
    end

    subgraph Parent Doc Flow
        PARENT_FLOW --> |embed query| PD1[Query Vector]
        PD1 --> |search child chunks| PD2[Find Child Chunks]
        VS --> PD2
        PD2 --> |lookup parent IDs| PD3[Retrieve Parent Docs]
        MEMSTORE[InMemoryStore] --> PD3
    end

    subgraph Compression Flow
        COMPRESS_FLOW --> |naive retrieval| CR1[Get k=10 Docs]
        VS --> CR1
        CR1 --> |rerank| CR2[Cohere Rerank Model]
        CR2 --> |top n relevant| CR3[Compressed Docs]
    end

    subgraph Ensemble Flow
        ENSEMBLE_FLOW --> |5 retrievers| EN1[Parallel Retrieval]
        VS --> EN1
        DOCS --> EN1
        EN1 --> |reciprocal rank fusion| EN2[Weighted Merge]
        EN2 --> |fused ranking| EN3[Ensemble Docs]
    end

    subgraph Semantic Flow
        DOCS --> |semantic chunking| SC0[Percentile Breakpoints]
        SC0 --> |embed chunks| SC1[Semantic Chunks]
        SEMANTIC_FLOW --> |embed query| SC2[Query Vector]
        SC2 --> |similarity search| SC3[Retrieved Semantic Chunks]
        SC1 --> SC3
    end

    NV3 --> MERGE[Context Aggregation]
    BM3 --> MERGE
    MQ3 --> MERGE
    PD3 --> MERGE
    CR3 --> MERGE
    EN3 --> MERGE
    SC3 --> MERGE

    MERGE --> |format context| PROMPT[Prompt Template<br/>Question + Context]
    PROMPT --> |generate| LLM[GPT-4.1-nano]
    LLM --> |stream| RESPONSE([Final Response])

    style START fill:#e1f5fe
    style RESPONSE fill:#e8f5e9
```

### Key Insights

1. **Parallel Strategy Pattern**: 7 independent retrieval flows process the same query in different ways
2. **Common Entry/Exit**: All flows start with a user query and end with the same LLM/prompt, isolating retrieval as the variable
3. **Embedding Duplication**: Naive, Multi-Query, Parent, and Semantic flows all embed the query independently (potential optimization)
4. **Two-Stage Patterns**: Compression (retrieve then rerank) and Parent Document (search children, return parents) use two-stage processing
5. **Data Reuse**: The same document corpus and embeddings are reused across multiple vector stores (memory-efficient)
6. **Context Aggregation**: All retrieved documents flow into a common context aggregation point before LLM generation

---

## Retrieval Strategy Patterns

This diagram illustrates the distinct retrieval strategies implemented in the project and their architectural patterns.

```mermaid
graph TB
    subgraph Strategy Taxonomy
        RETRIEVAL[Retrieval Strategies]
        RETRIEVAL --> DENSE[Dense Retrieval]
        RETRIEVAL --> SPARSE[Sparse Retrieval]
        RETRIEVAL --> HYBRID[Hybrid Methods]
        RETRIEVAL --> ENHANCED[Enhanced Retrieval]
    end

    subgraph Dense Vector-Based
        DENSE --> S1[Naive Retriever<br/>────────────<br/>Type: Dense Vector Search<br/>Method: Cosine Similarity<br/>k: 10<br/>Store: Qdrant<br/>────────────<br/>Pros: Simple, Fast<br/>Cons: Fixed embeddings]

        DENSE --> S7[Semantic Retriever<br/>────────────<br/>Type: Dense + Smart Chunking<br/>Method: Percentile Breakpoints<br/>k: 10<br/>Chunking: Semantic boundaries<br/>────────────<br/>Pros: Context-aware chunks<br/>Cons: Embedding overhead]
    end

    subgraph Sparse Keyword-Based
        SPARSE --> S2[BM25 Retriever<br/>────────────<br/>Type: Sparse Retrieval<br/>Method: Bag-of-Words TF-IDF<br/>Algorithm: Best-Matching 25<br/>────────────<br/>Pros: Exact keyword match<br/>Cons: No semantic understanding]
    end

    subgraph Query Enhancement
        ENHANCED --> S3[Multi-Query Retriever<br/>────────────<br/>Type: Query Expansion<br/>Method: LLM-generated variants<br/>Base: Naive Retriever<br/>LLM: GPT-4.1-nano<br/>────────────<br/>Pros: Better recall<br/>Cons: Higher latency/cost]
    end

    subgraph Hierarchical Methods
        ENHANCED --> S4[Parent Document Retriever<br/>────────────<br/>Type: Small-to-Big<br/>Method: Child search → Parent return<br/>Child Size: 750 chars<br/>Stores: Vector + InMemory<br/>────────────<br/>Pros: Precise + Contextual<br/>Cons: Complex setup]
    end

    subgraph Post-Processing
        ENHANCED --> S5[Compression Retriever<br/>────────────<br/>Type: Reranking<br/>Method: Cohere Rerank v3.5<br/>Base: Naive k=10<br/>Output: Top n compressed<br/>────────────<br/>Pros: High precision<br/>Cons: API cost]
    end

    subgraph Meta-Retrieval
        HYBRID --> S6[Ensemble Retriever<br/>────────────<br/>Type: Fusion<br/>Method: Reciprocal Rank Fusion<br/>Combines: 5 retrievers<br/>Weights: Equal 0.2 each<br/>────────────<br/>Pros: Robust performance<br/>Cons: Highest latency]
    end

    subgraph Comparison Framework
        S1 --> CHAIN[RAG Chain<br/>────────────<br/>Components:<br/>• Prompt Template<br/>• GPT-4.1-nano<br/>• StrOutputParser<br/>────────────<br/>Pattern: LCEL]
        S2 --> CHAIN
        S3 --> CHAIN
        S4 --> CHAIN
        S5 --> CHAIN
        S6 --> CHAIN
        S7 --> CHAIN

        CHAIN --> EVAL[Evaluation<br/>────────────<br/>Metrics:<br/>• Cost<br/>• Latency<br/>• Performance<br/>────────────<br/>Tool: Ragas + LangSmith]
    end

    style S1 fill:#e3f2fd
    style S2 fill:#fff3e0
    style S3 fill:#f3e5f5
    style S4 fill:#e8f5e9
    style S5 fill:#fce4ec
    style S6 fill:#e0f2f1
    style S7 fill:#f1f8e9
    style CHAIN fill:#ede7f6
    style EVAL fill:#fff9c4
```

### Key Insights

1. **Strategy Diversity**: The project implements 7 distinct retrieval patterns covering dense, sparse, hybrid, and enhanced methods
2. **Comparative Design**: All strategies feed into identical RAG chains, enabling fair performance comparison
3. **Trade-off Spectrum**:
   - **Speed**: BM25 (fastest) → Naive → Ensemble (slowest)
   - **Cost**: Naive (cheapest) → Compression/Multi-Query (most expensive)
   - **Precision**: Compression (highest) → Ensemble → Naive (variable)
4. **Composability**: Ensemble combines 5 strategies, showing meta-retrieval capabilities
5. **Evaluation-Driven**: The architecture is designed for empirical comparison using Ragas metrics and LangSmith observability
6. **Small-to-Big Pattern**: Parent Document Retriever implements a hierarchical retrieval strategy (search granular, return contextual)
7. **Query Expansion**: Multi-Query uses LLM to generate query variants, improving recall through diversity

---

## Implementation Insights

### Retrieval Strategy Characteristics

| Strategy | Type | Latency | Cost | Best Use Case |
|----------|------|---------|------|---------------|
| **Naive** | Dense Vector | Low | Low | Baseline semantic search |
| **BM25** | Sparse Keyword | Lowest | Lowest | Exact keyword matching |
| **Multi-Query** | Query Enhancement | High | High | Improving recall via query expansion |
| **Parent Document** | Hierarchical | Medium | Medium | Precise search with full context |
| **Compression** | Reranking | High | High | Maximizing precision |
| **Ensemble** | Meta-Retrieval | Highest | Highest | Robust performance across query types |
| **Semantic** | Smart Chunking | Medium | Medium | Documents with clear semantic boundaries |

### Architecture Principles

1. **Modularity**: Each retrieval strategy is independently implemented and testable
2. **Composability**: Strategies can be combined (as shown in Ensemble)
3. **Consistency**: Shared LLM, prompt, and evaluation framework across all strategies
4. **Observability**: LangSmith integration for tracing and cost analysis
5. **Extensibility**: New retrieval strategies can be added following the same pattern

### Key Files Reference

- **Configuration**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py` (lines 8-24)
- **Data Loading**: `adv-retrieval.py` (lines 59-80) / `session09-adv-retrieval.ipynb` (cell 5)
- **Vector Stores**: `session09-adv-retrieval.ipynb` (cells 9, 14, 18)
- **Retrievers**: `session09-adv-retrieval.ipynb` (cells 10, 22-25)
- **RAG Chains**: `session09-adv-retrieval.ipynb` (cells 28-40)
- **Evaluation**: `session09-adv-retrieval.ipynb` (cells 41-62)
