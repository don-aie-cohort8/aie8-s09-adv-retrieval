# Architecture Diagrams

## System Architecture

### Layered Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        NB1[Session 07: SDG & LangSmith]
        NB2[Session 08: RAGAS Evaluations]
        NB3[Session 09: Advanced Retrieval]
        ASSIGN[Assignment Notebook]
    end

    subgraph "Application Layer"
        CONFIG[Configuration Module<br/>config.py]
        SCRIPTS[Python Scripts<br/>session*.py]
    end

    subgraph "Business Logic Layer"
        RETR[Retrieval Strategies<br/>7 Implementations]
        EVAL[Evaluation Logic<br/>RAGAS & LangSmith]
        SDG[Synthetic Data Generation]
        CHAIN[RAG Chain Construction<br/>LCEL Pattern]
    end

    subgraph "Data Access Layer"
        VSTORE[Vector Stores<br/>Qdrant In-Memory]
        DSTORE[Document Stores<br/>InMemoryStore]
        LOADER[Document Loaders<br/>CSV & PDF]
    end

    subgraph "External Services"
        OPENAI[OpenAI API<br/>LLM & Embeddings]
        COHERE[Cohere API<br/>Rerank Model]
        LANGSMITH[LangSmith<br/>Tracing & Evaluation]
    end

    NB1 --> SCRIPTS
    NB2 --> SCRIPTS
    NB3 --> SCRIPTS
    ASSIGN --> SCRIPTS

    SCRIPTS --> CONFIG
    SCRIPTS --> RETR
    SCRIPTS --> EVAL
    SCRIPTS --> SDG

    RETR --> CHAIN
    EVAL --> CHAIN
    SDG --> CHAIN

    CHAIN --> VSTORE
    CHAIN --> DSTORE
    CHAIN --> LOADER

    VSTORE --> OPENAI
    CHAIN --> OPENAI
    CHAIN --> COHERE
    EVAL --> LANGSMITH
```

**Description**: This project follows a layered architecture pattern designed for educational exploration of advanced RAG retrieval strategies:

- **Presentation Layer**: Jupyter notebooks organized by session topic (sessions 07-09) providing interactive learning experiences
- **Application Layer**: Python script implementations of each session's concepts plus configuration management
- **Business Logic Layer**: Core RAG functionality including 7 retrieval strategies, evaluation frameworks (RAGAS/LangSmith), synthetic data generation, and LCEL chain patterns
- **Data Access Layer**: Abstraction for vector stores (Qdrant), document stores, and data loaders for CSV and PDF documents
- **External Services**: Integration with OpenAI (GPT-4.1 models, text-embedding-3-small/large), Cohere (rerank-v3.5), and LangSmith for observability

## Component Relationships

### Module Interaction

```mermaid
graph LR
    subgraph "Session Scripts"
        S07[session07-sdg-ragas-langsmith.py]
        S08[session08-ragas-rag-evals.py]
        S09[session09-adv-retrieval.py]
    end

    subgraph "Data Sources"
        CSV[Projects_with_Domains.csv]
        PDF[PDF Documents]
    end

    subgraph "Shared Components"
        EMB[OpenAIEmbeddings<br/>text-embedding-3-small]
        LLM[ChatOpenAI<br/>gpt-4.1-nano/mini]
        PROMPT[RAG_TEMPLATE<br/>Shared Prompt]
    end

    S07 --> PDF
    S08 --> PDF
    S09 --> CSV

    S07 --> EMB
    S08 --> EMB
    S09 --> EMB

    S07 --> LLM
    S08 --> LLM
    S09 --> LLM

    S08 --> PROMPT
    S09 --> PROMPT
```

**Description**: The modules interact through shared components and dependencies:

- **Session Scripts**: Each session script (07-09) is self-contained but follows common patterns
- **Configuration Module**: Provides centralized path management and default settings (chunk sizes, collection names)
- **Data Sources**: CSV data for project domains (sessions 09 & adv-retrieval), PDF documents for evaluation (sessions 07-08)
- **Shared Components**: Common embedding model (text-embedding-3-small), LLM instances (gpt-4.1 variants), and standardized RAG prompt template

### Retrieval Strategy Components

```mermaid
---
id: 1f9e88d6-1033-4670-8093-c0693894a768
---
graph TB
    subgraph "Input"
        QUERY[User Query]
        DOCS[Source Documents<br/>synthetic_usecase_data]
    end

    subgraph "7 Retrieval Strategies"
        NAIVE["Naive Retrieval<br/>Cosine Similarity<br/>k=10"]
        BM25["BM25 Retriever<br/>Sparse Bag-of-Words<br/>Keyword Matching"]
        COMPRESS["Contextual Compression<br/>Rerank with Cohere<br/>rerank-v3.5"]
        MULTI["Multi-Query Retriever<br/>LLM-Generated Variants<br/>Union of Results"]
        PARENT["Parent Document Retriever<br/>Small-to-Big Strategy<br/>750 char chunks"]
        ENSEMBLE["Ensemble Retriever<br/>Reciprocal Rank Fusion<br/>Equal Weights"]
        SEMANTIC["Semantic Chunking<br/>Percentile Threshold<br/>Semantic Similarity"]
    end

    subgraph "Vector Stores"
        VS1[Qdrant: Synthetic_Usecases<br/>1536 dimensions]
        VS2[Qdrant: full_documents<br/>Parent-Child]
        VS3[Qdrant: Semantic_Chunks<br/>Limited to 20 docs]
    end

    subgraph "RAG Chain Pattern"
        LCEL[LCEL Chain Structure<br/>context + question -> prompt -> LLM]
    end

    DOCS --> NAIVE
    DOCS --> BM25
    DOCS --> COMPRESS
    DOCS --> MULTI
    DOCS --> PARENT
    DOCS --> ENSEMBLE
    DOCS --> SEMANTIC

    NAIVE --> VS1
    COMPRESS --> VS1
    MULTI --> VS1
    PARENT --> VS2
    SEMANTIC --> VS3

    ENSEMBLE --> BM25
    ENSEMBLE --> NAIVE
    ENSEMBLE --> PARENT
    ENSEMBLE --> COMPRESS
    ENSEMBLE --> MULTI

    QUERY --> NAIVE
    QUERY --> BM25
    QUERY --> COMPRESS
    QUERY --> MULTI
    QUERY --> PARENT
    QUERY --> ENSEMBLE
    QUERY --> SEMANTIC

    NAIVE --> LCEL
    BM25 --> LCEL
    COMPRESS --> LCEL
    MULTI --> LCEL
    PARENT --> LCEL
    ENSEMBLE --> LCEL
    SEMANTIC --> LCEL
```

**Description**: The project implements 7 distinct retrieval strategies, each optimized for different scenarios:

1. **Naive Retrieval**: Baseline cosine similarity search returning top-10 documents
2. **BM25**: Classic sparse retrieval using bag-of-words for keyword-based matching
3. **Contextual Compression**: Retrieves 10 docs then reranks using Cohere's rerank-v3.5 model
4. **Multi-Query**: Generates multiple query variations via LLM, retrieves for each, returns union
5. **Parent Document Retriever**: Searches small chunks (750 chars) but returns full parent documents
6. **Ensemble Retriever**: Combines all 5 other retrievers using Reciprocal Rank Fusion with equal weights
7. **Semantic Chunking**: Pre-processing strategy that chunks by semantic similarity (percentile threshold)

All strategies feed into the same LCEL chain pattern for consistent evaluation.

## Class Hierarchies

### State Management Classes

```mermaid
classDiagram
    class State {
        <<TypedDict>>
        +str question
        +List~Document~ context
        +str response
    }

    class AdjustedState {
        <<TypedDict>>
        +str question
        +List~Document~ context
        +str response
    }

    class Document {
        <<LangChain Core>>
        +str page_content
        +dict metadata
    }

    class EvaluationDataset {
        <<RAGAS>>
        +to_pandas() DataFrame
        +to_csv(path) void
        +from_pandas(df) EvaluationDataset
    }

    class TestsetGenerator {
        <<RAGAS>>
        +LangchainLLMWrapper llm
        +LangchainEmbeddingsWrapper embedding_model
        +KnowledgeGraph knowledge_graph
        +generate(size, distribution) Testset
        +generate_with_langchain_docs(docs, size) Testset
    }

    class KnowledgeGraph {
        <<RAGAS>>
        +List~Node~ nodes
        +save(path) void
        +load(path) KnowledgeGraph
    }

    class Node {
        <<RAGAS>>
        +NodeType type
        +dict properties
    }

    State --> Document : contains
    AdjustedState --> Document : contains
    TestsetGenerator --> KnowledgeGraph : uses
    KnowledgeGraph --> Node : contains
```

**Description**: The project uses TypedDict classes for state management in LangGraph implementations:

- **State/AdjustedState**: TypedDict classes defining the state schema for RAG graphs (question, context documents, response)
- **Document**: LangChain core document class containing page_content and metadata
- **RAGAS Classes**: Evaluation framework classes for synthetic data generation (TestsetGenerator), knowledge graph construction (KnowledgeGraph, Node), and evaluation datasets (EvaluationDataset)

The state classes follow LangGraph patterns while RAGAS classes support the evaluation pipeline.

## Module Dependencies

### Import Graph

```mermaid
---
id: ca1e1523-9ce8-4c6e-b079-835bfc2cf908
---
graph TB
    subgraph "Session 09: Advanced Retrieval"
        S09[session09-adv-retrieval.py]
    end

    subgraph "Session 08: RAGAS Evals"
        S08[session08-ragas-rag-evals.py]
    end

    subgraph "Session 07: SDG & LangSmith"
        S07[session07-sdg-ragas-langsmith.py]
    end

    subgraph "LangChain Core"
        LC_CORE[langchain_core<br/>prompts, runnables,<br/>output_parsers]
        LC_RETR[langchain.retrievers<br/>Ensemble, Parent,<br/>Multi-Query, Compression]
        LC_SPLIT[langchain_text_splitters<br/>Recursive, Semantic]
    end

    subgraph "LangChain Community"
        LC_COMM[langchain_community<br/>BM25Retriever,<br/>CSVLoader, PDFLoader,<br/>Qdrant VectorStore]
    end

    subgraph "LangChain Integrations"
        LC_OAI[langchain_openai<br/>ChatOpenAI,<br/>OpenAIEmbeddings]
        LC_COH[langchain_cohere<br/>CohereRerank]
        LC_QDRANT[langchain_qdrant<br/>QdrantVectorStore]
    end

    subgraph "LangGraph"
        LG[langgraph.graph<br/>StateGraph, START]
    end

    subgraph "RAGAS_SG"["RAGAS"]
        RAGAS[ragas<br/>evaluate, metrics,<br/>TestsetGenerator,<br/>KnowledgeGraph]
    end

    subgraph "LangSmith"
        LS[langsmith<br/>Client, evaluate,<br/>LangChainStringEvaluator]
    end

    subgraph "Qdrant Client"
        QC[qdrant_client<br/>QdrantClient, models]
    end

    S09 --> LC_CORE
    S09 --> LC_RETR
    S09 --> LC_SPLIT
    S09 --> LC_COMM
    S09 --> LC_OAI
    S09 --> LC_COH
    S09 --> QC

    S08 --> LC_CORE
    S08 --> LC_RETR
    S08 --> LC_COMM
    S08 --> LC_OAI
    S08 --> LC_COH
    S08 --> LC_QDRANT
    S08 --> LG
    S08 --> RAGAS
    S08 --> QC

    S07 --> LC_CORE
    S07 --> LC_COMM
    S07 --> LC_OAI
    S07 --> RAGAS
    S07 --> LS
```

**Description**: The dependency graph shows clear separation of concerns:

- **Config Module**: Provides path configuration but has minimal dependencies (only pathlib)
- **Session 07**: Focuses on synthetic data generation (RAGAS) and evaluation (LangSmith)
- **Session 08**: Adds LangGraph for RAG pipeline orchestration and comprehensive RAGAS metrics
- **Session 09**: Concentrates on retrieval strategies with extensive LangChain retrievers
- **External Dependencies**: OpenAI (LLM & embeddings), Cohere (reranking), Qdrant (vector storage), RAGAS (evaluation), LangSmith (observability)

## Data Flow

### Retrieval Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Chain as RAG Chain (LCEL)
    participant Retriever
    participant VectorStore as Qdrant VectorStore
    participant Embeddings as OpenAI Embeddings
    participant Reranker as Cohere Rerank
    participant LLM as ChatOpenAI

    User->>Chain: invoke({"question": "query"})

    Note over Chain: Extract question via itemgetter

    Chain->>Retriever: invoke("query")

    alt Naive/BM25/Semantic/Multi-Query
        Retriever->>Embeddings: embed_query("query")
        Embeddings-->>Retriever: query_vector
        Retriever->>VectorStore: similarity_search(vector, k=10)
        VectorStore-->>Retriever: 10 documents
    else Parent Document
        Retriever->>Embeddings: embed_query("query")
        Embeddings-->>Retriever: query_vector
        Retriever->>VectorStore: search child chunks
        VectorStore-->>Retriever: child chunks
        Retriever->>Retriever: fetch parent documents
        Retriever-->>Retriever: parent documents
    else Contextual Compression
        Retriever->>VectorStore: similarity_search(k=10)
        VectorStore-->>Retriever: 10 documents
        Retriever->>Reranker: rerank(query, docs)
        Reranker-->>Retriever: top-N reranked docs
    else Ensemble
        Retriever->>Retriever: parallel retrieve from all
        Retriever->>Retriever: reciprocal rank fusion
    end

    Retriever-->>Chain: retrieved_contexts

    Note over Chain: Format prompt with<br/>question + context

    Chain->>LLM: invoke(formatted_prompt)
    LLM-->>Chain: response

    Chain-->>User: {"response": str, "context": List[Document]}
```

**Description**: The retrieval pipeline follows a consistent LCEL pattern across all strategies:

1. **Query Input**: User provides question as dict input
2. **Question Extraction**: LCEL chain extracts question via itemgetter
3. **Retrieval**: Strategy-specific retrieval logic executes:
   - **Vector-based**: Embed query, search vector store, return top-k
   - **Parent Document**: Search child chunks, return parent documents
   - **Compression**: Retrieve candidates, then rerank with Cohere
   - **Ensemble**: Parallel retrieval from multiple strategies, fuse with RRF
4. **Context Assembly**: Retrieved documents assigned to context key
5. **Prompt Formatting**: RAG template populated with question + context
6. **Generation**: LLM generates response based on formatted prompt
7. **Output**: Return both response and context for evaluation

### RAG Evaluation Flow

```mermaid
flowchart TD
    START([Start Evaluation]) --> LOAD_DATA[Load Source Documents<br/>CSV or PDF]

    LOAD_DATA --> SDG_CHOICE{Evaluation<br/>Approach?}

    SDG_CHOICE -->|Session 07/08| SDG[Synthetic Data Generation<br/>via RAGAS]
    SDG_CHOICE -->|Session 09| MANUAL[Manual Test Questions]

    SDG --> BUILD_KG[Build Knowledge Graph<br/>with default_transforms]
    BUILD_KG --> GENERATE[Generate Test Set<br/>Query Distribution:<br/>SingleHop: 50%<br/>MultiHopAbstract: 25%<br/>MultiHopSpecific: 25%]

    GENERATE --> GOLDEN[Golden Test Set<br/>user_input, reference,<br/>reference_contexts]

    MANUAL --> GOLDEN

    GOLDEN --> BUILD_CHAIN[Build RAG Chain<br/>Choose Strategy]

    BUILD_CHAIN --> VECTORIZE[Create Vector Store<br/>Split & Embed Documents]

    VECTORIZE --> SETUP_RETR[Configure Retriever<br/>Strategy-Specific Parameters]

    SETUP_RETR --> LCEL["Construct LCEL Chain<br/>retriever | prompt | LLM"]

    LCEL --> INFERENCE[Run Inference Loop<br/>For each test question]

    INFERENCE --> COLLECT[Collect Results<br/>response + retrieved_contexts]

    COLLECT --> EVAL_CHOICE{Evaluation<br/>Platform?}

    EVAL_CHOICE -->|RAGAS| RAGAS_EVAL[RAGAS Evaluation<br/>Metrics:<br/>- LLMContextRecall<br/>- Faithfulness<br/>- FactualCorrectness<br/>- ResponseRelevancy<br/>- ContextEntityRecall<br/>- NoiseSensitivity]

    EVAL_CHOICE -->|LangSmith| LS_EVAL[LangSmith Evaluation<br/>Evaluators:<br/>- QA Evaluator<br/>- Labeled Helpfulness<br/>- Custom Criteria]

    RAGAS_EVAL --> RESULTS[Evaluation Results<br/>Metrics Scores]
    LS_EVAL --> RESULTS

    RESULTS --> COMPARE[Compare Strategies<br/>Cost, Latency, Performance]

    COMPARE --> END([End Evaluation])

    style START fill:#90EE90
    style END fill:#FFB6C1
    style GOLDEN fill:#FFD700
    style RESULTS fill:#87CEEB
```

**Description**: The RAG evaluation workflow supports comprehensive testing and comparison:

1. **Data Loading**: Load source documents (CSV for session 09, PDF for sessions 07-08)

2. **Test Set Generation**:
   - **RAGAS Approach (Sessions 07-08)**: Build knowledge graph, apply transforms, generate synthetic test set with query distribution
   - **Manual Approach (Session 09)**: Use predefined questions for strategy comparison

3. **Golden Test Set**: Contains user_input (question), reference (expected answer), and reference_contexts (ground truth)

4. **RAG Chain Construction**:
   - Select retrieval strategy (1 of 7)
   - Split and embed documents
   - Configure strategy-specific parameters
   - Build LCEL chain pattern

5. **Inference**: Run each test question through the chain, collect response and retrieved contexts

6. **Evaluation Platform Selection**:
   - **RAGAS**: Comprehensive metrics including context recall, faithfulness, factual correctness, response relevancy, entity recall, and noise sensitivity
   - **LangSmith**: Custom evaluators for QA accuracy, helpfulness, and domain-specific criteria

7. **Results Analysis**: Compare strategies across cost (API calls), latency (execution time), and performance (metric scores)

## External Dependencies

### API and Service Integration

```mermaid
graph TB
    subgraph "Project Code"
        SCRIPTS[Python Scripts<br/>session*.py]
        NOTEBOOKS[Jupyter Notebooks<br/>*.ipynb]
    end

    subgraph "OpenAI Services"
        GPT4_NANO[GPT-4.1-nano<br/>Fast Generation<br/>Used in: RAG chains]
        GPT4_MINI[GPT-4.1-mini<br/>Balanced Performance<br/>Used in: Evaluation]
        GPT4[GPT-4.1<br/>High Quality<br/>Used in: SDG, Evaluation]
        EMB_SMALL[text-embedding-3-small<br/>1536 dimensions<br/>Used in: Vector stores]
        EMB_LARGE[text-embedding-3-large<br/>Used in: Session 07]
    end

    subgraph "Cohere Services"
        RERANK[rerank-v3.5<br/>Contextual Compression<br/>Reranking Model]
    end

    subgraph "LangSmith Services"
        LS_TRACE[Tracing & Logging<br/>LANGCHAIN_TRACING_V2]
        LS_EVAL[Evaluation Platform<br/>Dataset Management<br/>Custom Evaluators]
        LS_PROJ[Project Organization<br/>Session-based Projects]
    end

    subgraph "Qdrant"
        QDRANT_MEM[In-Memory Vector Store<br/>Collections:<br/>- Synthetic_Usecases<br/>- full_documents<br/>- Semantic_Chunks<br/>- Use Case RAG]
    end

    subgraph "RAGAS Framework"
        RAGAS_SDG[Synthetic Data Generation<br/>TestsetGenerator]
        RAGAS_METRICS[Evaluation Metrics<br/>6 Core Metrics]
        RAGAS_KG[Knowledge Graph<br/>Transform Pipeline]
    end

    subgraph "Data Sources"
        CSV_DATA[Projects_with_Domains.csv<br/>Project metadata & descriptions]
        PDF_DATA[PDF Documents<br/>Use case documentation]
    end

    SCRIPTS --> GPT4_NANO
    SCRIPTS --> GPT4_MINI
    SCRIPTS --> GPT4
    SCRIPTS --> EMB_SMALL
    NOTEBOOKS --> EMB_LARGE

    SCRIPTS --> RERANK

    SCRIPTS --> LS_TRACE
    SCRIPTS --> LS_EVAL
    SCRIPTS --> LS_PROJ

    SCRIPTS --> QDRANT_MEM

    SCRIPTS --> RAGAS_SDG
    SCRIPTS --> RAGAS_METRICS
    SCRIPTS --> RAGAS_KG

    SCRIPTS --> CSV_DATA
    SCRIPTS --> PDF_DATA

    style GPT4_NANO fill:#74AA9C
    style EMB_SMALL fill:#74AA9C
    style RERANK fill:#D4A5A5
    style LS_TRACE fill:#9B9ECE
    style QDRANT_MEM fill:#FFD93D
    style RAGAS_SDG fill:#95E1D3
    style CSV_DATA fill:#F8B400
```

**Description**: The project integrates with multiple external services and frameworks:

### OpenAI API Integration
- **GPT-4.1-nano**: Lightweight model for RAG response generation in all retrieval chains
- **GPT-4.1-mini**: Mid-tier model for RAGAS evaluations and balanced tasks
- **GPT-4.1**: Premium model for synthetic data generation and high-quality evaluation
- **text-embedding-3-small**: Primary embedding model (1536-dim) for all vector stores
- **text-embedding-3-large**: Alternative embedding model used in session 07

### Cohere API Integration
- **rerank-v3.5**: State-of-the-art reranking model for contextual compression retrieval strategy
- Compresses initial retrieval results to top-N most relevant documents

### LangSmith Integration
- **Tracing**: Automatic logging of all LangChain executions via LANGCHAIN_TRACING_V2
- **Evaluation Platform**: Dataset creation, management, and custom evaluators
- **Project Organization**: Session-based project naming with UUID for isolation

### Qdrant Vector Database
- **In-Memory Collections**: Four distinct collections for different strategies
  - `Synthetic_Usecases`: Main collection for naive/BM25/compression/multi-query
  - `full_documents`: Parent-child structure for parent document retrieval
  - `Semantic_Chunks`: Semantically chunked documents
  - `Use Case RAG`: Collection for session 07-08 evaluations

### RAGAS Framework
- **Synthetic Data Generation**: TestsetGenerator with knowledge graph construction
- **Evaluation Metrics**: 6 core metrics (context recall, faithfulness, factual correctness, response relevancy, entity recall, noise sensitivity)
- **Knowledge Graph**: Document transformation and relationship extraction

### Data Sources
- **CSV Data**: Projects_with_Domains.csv containing project metadata, domains, descriptions, and judge scores
- **PDF Documents**: Use case documentation for evaluation workflows

All external integrations require API keys configured via environment variables (OPENAI_API_KEY, COHERE_API_KEY, LANGCHAIN_API_KEY).
