# Repository Architecture Documentation

> **Generated**: October 8, 2025
> **Project**: Advanced Retrieval Strategies for RAG (Retrieval-Augmented Generation)

## Overview

### Project Purpose

This educational repository demonstrates **advanced retrieval techniques for RAG (Retrieval-Augmented Generation) systems** using the LangChain ecosystem. The project provides a comprehensive exploration of seven distinct retrieval strategies, synthetic data generation with RAGAS, and evaluation frameworks using both RAGAS metrics and LangSmith integration.

The codebase is structured as a progressive learning journey through three main sessions, each building upon previous concepts to create a complete understanding of modern RAG system architecture and evaluation.

### Educational Focus

This is an **educational and tutorial project** designed for:
- AI Engineering students learning advanced RAG techniques
- Researchers exploring retrieval strategy performance
- Developers seeking practical implementations of LangChain patterns
- Teams evaluating different retrieval approaches for production systems

The project emphasizes:
- **Hands-on learning** through Jupyter notebooks and Python scripts
- **Comparative analysis** of retrieval strategies with real metrics
- **Best practices** in RAG system design and evaluation
- **LangChain Expression Language (LCEL)** patterns for production-ready code

### Key Capabilities

- **Synthetic Data Generation**: Create high-quality test datasets using RAGAS knowledge graphs with multiple query types (single-hop, multi-hop abstract, multi-hop specific)
- **Seven Retrieval Strategies**: Naive, BM25, Contextual Compression, Multi-Query, Parent Document, Ensemble, and Semantic Chunking
- **Comprehensive Evaluation**: Six RAGAS metrics (LLM Context Recall, Faithfulness, Factual Correctness, Response Relevancy, Context Entity Recall, Noise Sensitivity)
- **LangSmith Integration**: Full tracing, monitoring, and custom evaluation capabilities
- **LangGraph Workflows**: State-managed RAG pipelines with retrieve → generate patterns
- **Production Patterns**: LCEL chain construction for consistent, traceable implementations

## Quick Start

### For New Developers

1. **Start with the Component Inventory** - Read [01_component_inventory.md](docs/01_component_inventory.md) to understand what's in the codebase and where everything is located
2. **Review Architecture Diagrams** - Study [02_architecture_diagrams.md](diagrams/02_architecture_diagrams.md) to see how components fit together in a layered architecture
3. **Understand Data Flows** - Follow [03_data_flows.md](docs/03_data_flows.md) to see exactly how data moves through each retrieval strategy
4. **Explore API Reference** - Use [04_api_reference.md](docs/04_api_reference.md) as your implementation guide with detailed examples

**Recommended Learning Path**:
- Begin with Session 09 notebooks to see retrieval strategies in action
- Move to Session 07 to understand synthetic data generation
- Complete with Session 08 to learn evaluation frameworks
- Study the consolidated scripts for production-ready patterns

### For Researchers/Students

This project is ideal for understanding:
- **How retrieval strategies differ**: Compare semantic search, sparse retrieval, reranking, query expansion, and hybrid approaches
- **Performance trade-offs**: Analyze latency, cost, and quality metrics across strategies
- **Evaluation methodologies**: Learn RAGAS metrics and LangSmith evaluation workflows
- **Best practices**: Discover when to use each retrieval strategy based on use case requirements

**Key Learning Questions**:
- When should I use BM25 vs. semantic search?
- How does reranking improve retrieval quality?
- What's the cost-benefit of ensemble retrieval?
- How do I evaluate RAG systems objectively?

### Documentation Map

- **[Component Inventory](docs/01_component_inventory.md)** - Complete catalog of modules, classes, and functions with file references and line numbers
- **[Architecture Diagrams](diagrams/02_architecture_diagrams.md)** - Visual representation of system structure including layered architecture, component relationships, and data flows
- **[Data Flow Analysis](docs/03_data_flows.md)** - Sequence diagrams showing data movement through all seven retrieval strategies, evaluation workflows, and LangSmith integration
- **[API Reference](docs/04_api_reference.md)** - Detailed API documentation with examples, usage patterns, and best practices

## Architecture Summary

### System Layers

The project follows a **layered architecture pattern** designed for educational exploration:

**Presentation Layer**: Jupyter notebooks organized by session topic (sessions 07-09) providing interactive learning experiences

**Application Layer**: Python script implementations of each session's concepts plus configuration management

**Business Logic Layer**: Core RAG functionality including:
- 7 retrieval strategies (Naive, BM25, Contextual Compression, Multi-Query, Parent Document, Ensemble, Semantic Chunking)
- Evaluation frameworks (RAGAS metrics, LangSmith evaluators)
- Synthetic data generation with knowledge graphs
- LCEL chain patterns for consistent implementation

**Data Access Layer**: Abstraction for vector stores (Qdrant), document stores (InMemoryStore), and data loaders for CSV and PDF documents

**External Services**: Integration with OpenAI (GPT-4.1 models, text-embedding-3-small/large), Cohere (rerank-v3.5), and LangSmith for observability

### Three-Session Structure

The project progresses through three educational sessions, each building on previous concepts:

#### Session 07: Synthetic Data Generation

- **Key Capabilities**: RAGAS knowledge graph construction, synthetic test set generation, LangSmith dataset creation
- **Main Components**:
  - `TestsetGenerator` with multiple query synthesizers (SingleHop, MultiHopAbstract, MultiHopSpecific)
  - Knowledge graph with document transforms for entity/relationship extraction
  - LangSmith evaluators (QA, helpfulness, custom criteria)
- **Output**: Golden test datasets with user_input, reference answers, and reference_contexts
- **File**: `session07-sdg-ragas-langsmith.py` (333 lines)

#### Session 08: RAG Evaluation

- **Key Capabilities**: LangGraph workflow construction, baseline vs. reranked retrieval comparison, comprehensive RAGAS evaluation
- **Main Components**:
  - `State` and `AdjustedState` TypedDict for graph state management
  - `retrieve()` and `generate()` node functions
  - `retrieve_reranked()` with Cohere rerank-v3.5
  - LangGraph compilation and execution
- **Metrics Used**: LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
- **File**: `session08-ragas-rag-evals.py` (303 lines)

#### Session 09: Advanced Retrieval Strategies

- **Key Capabilities**: Seven retrieval strategy implementations, LCEL chain construction, performance comparison
- **Seven Strategies Implemented**:
  1. **Naive Retrieval** - Cosine similarity search (k=10)
  2. **BM25** - Sparse retrieval with keyword matching
  3. **Contextual Compression** - Reranking with Cohere rerank-v3.5
  4. **Multi-Query** - LLM-generated query variants with result fusion
  5. **Parent Document** - Small-to-big retrieval (750-char chunks → full docs)
  6. **Ensemble** - Reciprocal Rank Fusion combining all strategies
  7. **Semantic Chunking** - Percentile-based semantic boundary detection
- **Comparison Approach**: Consistent LCEL chain pattern across all strategies for fair evaluation
- **File**: `session09-adv-retrieval.py` (327 lines)

### Architecture Patterns

**Key Patterns Identified**:

1. **LCEL Chain Pattern** - All retrieval strategies use consistent LangChain Expression Language structure:
   ```
   {context + question} → RunnablePassthrough → {response + context}
   ```

2. **Retrieve → Generate Pattern** - Core RAG workflow across all implementations:
   ```
   Question → Retriever → Context → Prompt + LLM → Response
   ```

3. **State Management Pattern** - LangGraph TypedDict states flowing through node functions:
   ```
   State {question, context, response} → retrieve() → generate() → Final State
   ```

4. **Two-Stage Retrieval Pattern** - Over-retrieve then rerank/compress for quality:
   ```
   Retrieve(k=20) → Rerank/Compress → Top-N → Generation
   ```

5. **Small-to-Big Pattern** - Parent document retrieval searching small chunks, returning large context:
   ```
   Search(child_chunks) → Fetch(parent_ids) → Return(full_documents)
   ```

6. **Fusion Pattern** - Ensemble retrieval combining multiple strategies with RRF:
   ```
   Parallel(retriever1, retriever2, ..., retrieverN) → RRF → Merged Results
   ```

## Component Overview

### Project Structure

```
aie8-s09-adv-retrieval/
├── config.py                                    # Configuration management
├── session07-sdg-ragas-langsmith.py            # Synthetic data generation
├── session08-ragas-rag-evals.py                # RAG evaluation with RAGAS
├── session09-adv-retrieval.py                  # Seven retrieval strategies
├── adv-retrieval.py                            # Full tutorial version
├── data/
│   ├── Projects_with_Domains.csv              # Project metadata dataset
│   └── howpeopleuseai.pdf                      # PDF documents for testing
├── notebooks/
│   ├── session07-sdg-ragas-langsmith.ipynb
│   ├── session08-ragas-rag-evals.ipynb
│   └── session09-adv-retrieval.ipynb
└── docs/
    └── [project documentation]
```

### Core Modules

#### config.py

- **Purpose**: Central configuration management for project paths and settings
- **Key Components**:
  - Path management: `PROJECT_ROOT`, `DATA_DIR`, `NOTEBOOKS_DIR`, `DOCS_DIR`
  - Default settings: `DEFAULT_CHUNK_SIZE=500`, `DEFAULT_CHUNK_OVERLAP=50`
  - Qdrant configuration: `QDRANT_COLLECTION_NAME`, `QDRANT_LOCATION=":memory:"`
- **Reference**: See [API Reference](docs/04_api_reference.md#configuration-module)
- **Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`

#### session07-sdg-ragas-langsmith.py

- **Purpose**: Demonstrates synthetic data generation using RAGAS knowledge graphs and LangSmith evaluation
- **Key Components**:
  - Knowledge graph construction with `KnowledgeGraph` and `default_transforms`
  - Test set generation with `TestsetGenerator` using three query synthesizers
  - LangSmith dataset creation and custom evaluators (QA, helpfulness, dopeness)
  - Baseline vs. Dope RAG chain comparison
- **Reference**: See [API Reference](docs/04_api_reference.md#session-07-synthetic-data-generation)
- **Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session07-sdg-ragas-langsmith.py`

#### session08-ragas-rag-evals.py

- **Purpose**: RAG evaluation using RAGAS metrics with LangGraph state management
- **Key Components**:
  - `State` and `AdjustedState` TypedDict classes for state management
  - `retrieve()`, `generate()`, `retrieve_reranked()` node functions
  - LangGraph baseline and rerank graph construction
  - Six RAGAS metrics evaluation (context recall, faithfulness, factual correctness, relevancy, entity recall, noise sensitivity)
- **Reference**: See [API Reference](docs/04_api_reference.md#session-08-rag-evaluation)
- **Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session08-ragas-rag-evals.py`

#### session09-adv-retrieval.py

- **Purpose**: Consolidated implementation of seven advanced retrieval strategies
- **Key Components**:
  - Seven retrieval strategies: Naive, BM25, Contextual Compression, Multi-Query, Parent Document, Ensemble, Semantic Chunking
  - Consistent LCEL chain pattern across all strategies
  - Vector store management with Qdrant
  - Sample invocations for each strategy
- **Reference**: See [API Reference](docs/04_api_reference.md#session-09-advanced-retrieval-strategies)
- **Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session09-adv-retrieval.py`

### Key Classes and Functions

- **State (TypedDict)** - State schema for baseline RAG graph with fields: question, context, response | Location: `session08-ragas-rag-evals.py:157-160`

- **AdjustedState (TypedDict)** - State schema for reranked RAG graph (identical to State) | Location: `session08-ragas-rag-evals.py:204-207`

- **retrieve(state)** - Retrieval node function for baseline graph, invokes retriever with question from state | Location: `session08-ragas-rag-evals.py:145-147`

- **generate(state)** - Generation node function, formats context and question into prompt, invokes LLM | Location: `session08-ragas-rag-evals.py:150-154`

- **retrieve_reranked(state)** - Retrieval node with Cohere reranking (k=20 → top-5) | Location: `session08-ragas-rag-evals.py:195-201`

## Retrieval Strategies

### Overview

The project implements **seven distinct retrieval strategies**, each optimized for different scenarios and use cases. All strategies are implemented with a consistent LCEL chain pattern for fair comparison and easy integration.

### Strategy Comparison

| Strategy | Type | Use Case | Latency | Quality | Complexity |
|----------|------|----------|---------|---------|------------|
| 1. Naive | Dense Vector | General purpose, semantic search | Low (~100-200ms) | Baseline | Low |
| 2. BM25 | Sparse/Keyword | Exact keyword matching, terminology | Low (~100-200ms) | Good for keywords | Low |
| 3. Contextual Compression | Dense + Rerank | High precision, quality over speed | High (~500-800ms) | Excellent | Medium |
| 4. Multi-Query | Dense + LLM | Complex queries, improved recall | Medium (~500-1000ms) | Very Good | Medium |
| 5. Parent Document | Dense (Small-to-Big) | Precise search + broad context | Medium (~200-400ms) | Very Good | Medium-High |
| 6. Ensemble | Hybrid (All Methods) | Production robustness | Very High (~1-2s) | Excellent | High |
| 7. Semantic Chunking | Dense (Semantic Split) | Coherent document chunks | Low (~100-300ms) | Good | Medium |

### Detailed Strategy Information

1. **Naive Retrieval** - Standard cosine similarity search using OpenAI embeddings (text-embedding-3-small) with k=10. Best for general-purpose retrieval with clear semantic patterns. → [Details](docs/04_api_reference.md#1-naive-retrieval-cosine-similarity)

2. **BM25** - Classic sparse retrieval using Best Matching 25 algorithm for term-based ranking. Excellent for keyword-heavy queries and exact term matching without embedding overhead. → [Details](docs/04_api_reference.md#2-bm25-sparse-retrieval)

3. **Contextual Compression** - Two-stage approach: retrieve 10 candidates via naive search, then rerank using Cohere rerank-v3.5. Provides highest precision by filtering irrelevant documents. → [Details](docs/04_api_reference.md#3-contextual-compression-reranking)

4. **Multi-Query** - Generates 3-5 query variations using LLM, retrieves for each variant, then returns union of results. Improves recall by capturing different query interpretations. → [Details](docs/04_api_reference.md#4-multi-query-retrieval)

5. **Parent Document** - Small-to-big strategy: searches 750-char child chunks for precision, returns full parent documents for context. Balances accurate retrieval with comprehensive information. → [Details](docs/04_api_reference.md#5-parent-document-retrieval)

6. **Ensemble** - Combines five retrieval strategies (BM25, Naive, Parent Document, Compression, Multi-Query) using Reciprocal Rank Fusion with equal weights. Most robust but slowest. → [Details](docs/04_api_reference.md#6-ensemble-retrieval)

7. **Semantic Chunking** - Creates variable-sized chunks based on semantic coherence using percentile-based similarity thresholds. Preserves topically coherent segments. → [Details](docs/04_api_reference.md#7-semantic-chunking)

## Data Flows

### Key Flow Patterns

The project demonstrates three primary data flow patterns:

1. **Synthetic Data Generation** - PDF documents → Knowledge graph → Query synthesis → Golden test sets
2. **RAG Evaluation** - Test questions → Retrieval → Generation → RAGAS metrics → Evaluation results
3. **Retrieval Strategies** - Question → Strategy-specific retrieval → Context → LLM generation → Response

All flows follow the core RAG pattern: **Question → Retrieval → Context → Generation → Response**

### Synthetic Data Generation Flow

**Overview**: Transforms raw PDF documents into structured knowledge graphs, then synthesizes diverse test questions with multiple difficulty levels.

- **Input**: PDF documents from data directory
- **Process**:
  1. Load documents with `DirectoryLoader` + `PyMuPDFLoader`
  2. Build `KnowledgeGraph` with document nodes
  3. Apply `default_transforms` for entity/relationship extraction via LLM
  4. Generate synthetic test set with `TestsetGenerator` using three query types (50% SingleHop, 25% MultiHopAbstract, 25% MultiHopSpecific)
  5. Create LangSmith dataset with inputs, outputs, and metadata
- **Output**: Golden testset with user_input, reference answers, reference_contexts
- **Details**: See [Data Flow Analysis](docs/03_data_flows.md#synthetic-data-generation-flow-session-07)

### RAG Evaluation Flow

**Overview**: Evaluates RAG systems using LangGraph workflows and comprehensive RAGAS metrics, comparing baseline vs. reranked retrieval.

- **Input**: Golden testset from synthetic data generation
- **Process**:
  1. Split documents into 500-char chunks with overlap
  2. Create Qdrant vector store with OpenAI embeddings
  3. Build LangGraph with retrieve → generate nodes
  4. For each test question: invoke graph, collect response + contexts
  5. Run RAGAS evaluation with 6 metrics
- **Output**: Evaluation results with scores for context recall, faithfulness, factual correctness, relevancy, entity recall, noise sensitivity
- **Details**: See [Data Flow Analysis](docs/03_data_flows.md#rag-evaluation-flow-session-08)

### Retrieval Strategy Flows

**Overview**: Each of the seven retrieval strategies implements a unique approach to finding relevant documents, all using the consistent LCEL chain pattern.

- **Common Pattern**: All strategies follow the LCEL chain structure:
  ```
  {context: question | retriever, question: question}
  → RunnablePassthrough.assign(context)
  → {response: prompt | LLM, context: context}
  ```

- **Variations**: Each strategy differs only in the retriever implementation:
  - **Naive**: Direct vector similarity search (k=10)
  - **BM25**: Inverted index with term matching
  - **Compression**: Naive retrieval (k=10) → Cohere reranking
  - **Multi-Query**: LLM query expansion → parallel retrieval → fusion
  - **Parent Document**: Child chunk search → parent document return
  - **Ensemble**: Parallel execution of 5 retrievers → RRF fusion
  - **Semantic**: Semantic chunking → vector search

- **Details**: See [Data Flow Analysis](docs/03_data_flows.md#advanced-retrieval-flows-session-09)

### State Management

**LangGraph State Pattern**: Uses TypedDict classes (State, AdjustedState) to manage data flow through graph nodes.

State transitions: `Initial {question}` → `retrieve() adds {context}` → `generate() adds {response}` → `Final {question, context, response}`

- **Details**: See [Data Flow Analysis](docs/03_data_flows.md#data-state-transitions)

## Evaluation Framework

### RAGAS Metrics

The project uses six RAGAS metrics for comprehensive RAG evaluation:

1. **LLM Context Recall** - Measures if all ground truth information is present in retrieved context. Evaluates recall quality of retrieval step.

2. **Faithfulness** - Measures if the generated answer is faithful to the retrieved context. Detects hallucinations and ensures grounding.

3. **Factual Correctness** - Evaluates factual accuracy of the generated answer against reference. Measures generation quality.

4. **Semantic Similarity** - Computes semantic similarity between generated response and reference answer using embeddings.

5. **LLM Context Precision** - Measures precision of retrieved context. Identifies irrelevant documents in retrieval results.

6. **LLM Context Utilization** - Evaluates how well the LLM utilizes the provided context in generating responses.

**Details**: See [API Reference](docs/04_api_reference.md#ragas-evaluation)

### LangSmith Integration

LangSmith provides distributed tracing, evaluation platform, and dataset management:

- **Tracing**: Automatic logging of all LangChain executions with hierarchical spans
- **Captured Metrics**: Latency (retrieval, LLM, total), tokens (prompt, completion, total), cost (estimated per call), quality (evaluator scores)
- **Custom Evaluators**: QA evaluator (correctness), labeled helpfulness (with reference), custom criteria (dopeness, creativity)
- **Dataset Management**: Create, store, and version test datasets with inputs/outputs/metadata
- **Evaluation Platform**: Run evaluate() with chains and evaluators, view results in UI with comparisons and trends

**Configuration**:
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = f"My-Project-{uuid}"
```

## External Dependencies

### APIs and Services

- **OpenAI** - LLM and embeddings provider
  - Models used: gpt-4.1-nano (fast generation), gpt-4.1-mini (balanced), gpt-4.1 (high quality)
  - Embeddings: text-embedding-3-small (1536-dim, primary), text-embedding-3-large (alternative)
  - Purpose: RAG response generation, query expansion, evaluation, embeddings for vector search

- **Cohere** - Reranking service
  - Model used: rerank-v3.5 (state-of-art cross-encoder)
  - Purpose: Contextual compression retrieval, improving precision through reranking

- **LangSmith** - Observability and evaluation platform
  - Features used: Distributed tracing (LANGCHAIN_TRACING_V2), evaluation platform, dataset management, custom evaluators
  - Purpose: Debugging, cost tracking, performance monitoring, RAG evaluation workflows

- **Qdrant** - Vector database
  - Configuration: In-memory storage (`:memory:`), COSINE distance metric
  - Collections: Synthetic_Usecases, full_documents, Semantic_Chunks, Use Case RAG
  - Purpose: Document embedding storage, similarity search, parent-child retrieval

### Key Python Packages

**LangChain Ecosystem**:
- `langchain>=0.3.14` - Core RAG framework
- `langchain-openai>=0.3.3` - OpenAI integration
- `langchain-community>=0.3.16` - Community integrations (BM25, document loaders)
- `langchain-cohere==0.4.4` - Cohere reranking
- `langchain-qdrant>=0.2.0` - Qdrant vector store
- `langgraph>=0.2.69` - Graph-based workflows

**Evaluation & Processing**:
- `ragas==0.2.10` - RAG evaluation framework
- `pymupdf>=1.24.0` - PDF document loading
- `qdrant-client>=1.7.0` - Qdrant Python client

**Complete List**: See [API Reference](docs/04_api_reference.md#dependencies)

## Configuration

### Environment Variables Required

```bash
# Required for all sessions
OPENAI_API_KEY=your_openai_key_here

# Required for sessions 08 & 09
COHERE_API_KEY=your_cohere_key_here

# Required for LangSmith tracing and evaluation
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Your-Project-Name
```

### Configuration Files

- **config.py** - Central configuration for paths and settings
  - Path configuration: PROJECT_ROOT, DATA_DIR, NOTEBOOKS_DIR, DOCS_DIR
  - Model defaults: DEFAULT_CHUNK_SIZE=500, DEFAULT_CHUNK_OVERLAP=50
  - Qdrant settings: QDRANT_COLLECTION_NAME="Use Case RAG", QDRANT_LOCATION=":memory:"

**Details**: See [API Reference](docs/04_api_reference.md#configuration-module)

## Getting Started

### Prerequisites

- Python 3.9+
- API keys for OpenAI, Cohere, and LangSmith
- 8GB+ RAM recommended for vector operations
- Basic understanding of RAG concepts and LangChain

### Installation

```bash
# Clone repository
cd aie8-s09-adv-retrieval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key"
export COHERE_API_KEY="your_key"
export LANGSMITH_API_KEY="your_key"
```

### Running the Sessions

#### Session 07: Generate Synthetic Data

```bash
python session07-sdg-ragas-langsmith.py
```

**What it does**:
- Loads PDF documents from data directory
- Builds knowledge graph with entity/relationship extraction
- Generates 10 synthetic test questions (50% single-hop, 25% multi-hop abstract, 25% multi-hop specific)
- Creates LangSmith dataset with golden testset
- Evaluates baseline vs. dope RAG chains with custom criteria

#### Session 08: Evaluate RAG

```bash
python session08-ragas-rag-evals.py
```

**What it does**:
- Splits documents into 500-char chunks
- Creates baseline and reranked retrieval graphs with LangGraph
- Generates synthetic testset with RAGAS
- Runs both graphs on test questions
- Evaluates with 6 RAGAS metrics (context recall, faithfulness, factual correctness, relevancy, entity recall, noise sensitivity)
- Compares baseline vs. reranked performance

#### Session 09: Compare Retrieval Strategies

```bash
python session09-adv-retrieval.py
```

**What it does**:
- Loads CSV project data
- Implements seven retrieval strategies with consistent LCEL pattern
- Creates Qdrant vector stores for each strategy
- Provides sample invocations for testing
- Enables performance comparison across strategies

### Using Jupyter Notebooks

All sessions have corresponding Jupyter notebooks for interactive exploration:

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks directory
# Open: session07-sdg-ragas-langsmith.ipynb
# Open: session08-ragas-rag-evals.ipynb
# Open: session09-adv-retrieval.ipynb
```

Notebooks include:
- Detailed markdown explanations
- Step-by-step code execution
- Visualization of results
- Discussion questions and breakout activities

## Usage Examples

### Quick Example: Running a Retrieval Strategy

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chat_model = ChatOpenAI(model="gpt-4.1-nano")

# Create vector store
vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="MyCollection"
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Define prompt
RAG_TEMPLATE = """Use the context below to answer the question.
Context: {context}
Question: {question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Build LCEL chain
retrieval_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# Invoke
result = retrieval_chain.invoke({"question": "What is AI?"})
print(result["response"].content)
```

### Quick Example: Evaluating with RAGAS

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Prepare evaluation dataset
# (Assumes you have responses and retrieved_contexts populated)
eval_dataset = EvaluationDataset.from_pandas(dataset_df)

# Configure evaluator
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

# Run evaluation
result = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(),
        FactualCorrectness()
    ],
    llm=evaluator_llm
)

print(result)
```

## Performance Considerations

### Latency by Strategy

**Approximate single-query latency**:

- **Fastest (100-200ms)**: Naive Retrieval, BM25
- **Fast (100-300ms)**: Semantic Chunking (once chunks created)
- **Medium (200-400ms)**: Parent Document Retrieval
- **Slow (500-1000ms)**: Multi-Query Retrieval, Contextual Compression
- **Very Slow (1-2s)**: Ensemble Retrieval

**Factors affecting latency**: Network latency to APIs, corpus size, embedding model, k parameter, LLM model speed

### Quality vs. Speed Trade-offs

**High Precision Strategies** (Quality over Speed):
- Contextual Compression with reranking
- Ensemble retrieval combining multiple methods
- Trade-off: +15-20% quality improvement, +300-500ms latency

**High Recall Strategies** (Coverage over Precision):
- Multi-Query with query expansion
- Parent Document with full context
- Trade-off: More comprehensive results, potential noise increase

**Balanced Strategies**:
- Naive retrieval for general purpose
- BM25 for keyword-heavy queries
- Semantic chunking for coherent results

**Recommended by Use Case**:
- **Accuracy Critical**: Ensemble + Reranking (slow but best quality)
- **Speed Critical**: Naive or BM25 (fast but lower quality)
- **Balanced**: Contextual Compression (good quality, moderate speed)
- **Long-form Context**: Parent Document (preserves coherence)
- **Keyword-Heavy**: BM25 or BM25 + Ensemble (exact matching)

### Resource Usage

**Token Consumption**:
- Naive (k=10): ~5000 tokens context
- Reranked (top-5): ~2500 tokens context (50% reduction)
- Parent Document: 2000+ tokens per document (2-3× naive)
- Multi-Query: +500-1000 tokens per query variant

**Cost Optimization**:
- Use gpt-4.1-nano for generation ($0.15/1M input tokens)
- Use text-embedding-3-small ($0.02/1M tokens)
- Aggressive reranking: retrieve many, return few
- Cache embeddings when possible

**Memory Considerations**:
- In-memory Qdrant: ~1-2 MB per 1000 documents
- Parent document InMemoryStore: Can consume GBs for large corpora
- Production: Use persistent storage (disk Qdrant, Redis docstore)

**Details**: See [API Reference](docs/04_api_reference.md#performance-considerations)

## Best Practices

### Strategy Selection

**When to use each retrieval strategy**:

- **Naive Retrieval**: Baseline, general purpose, semantically well-structured documents, speed critical
- **BM25**: Keyword queries, specific terminology, exact matching important, no embedding infrastructure
- **Contextual Compression**: Quality over speed, refining initial retrieval, complex nuanced queries
- **Multi-Query**: Ambiguous queries, improving recall, vocabulary mismatch, query interpretation variety
- **Parent Document**: Hierarchical documents, precise search + broad context, avoiding information loss
- **Ensemble**: Production robustness, different query types, hedge against strategy weaknesses, quality critical
- **Semantic Chunking**: Documents with clear semantic structure, chunk quality critical, slower preprocessing acceptable

### Configuration Tips

**Optimal settings for different use cases**:

```python
# High-precision, cost-aware
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = CohereRerank(model="rerank-v3.5", top_n=3)

# High-recall, comprehensive
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
    llm=chat_model
)

# Balanced production
ensemble = EnsembleRetriever(
    retrievers=[bm25, naive, compression],
    weights=[0.3, 0.3, 0.4]  # Favor reranked
)
```

### Evaluation Workflow

**How to effectively evaluate retrieval strategies**:

1. **Generate Golden Testset**: Use RAGAS synthetic data generation with diverse query types
2. **Prepare Evaluation Dataset**: Run each strategy on testset, collect responses and contexts
3. **Configure RAGAS Metrics**: Select appropriate metrics (context recall, faithfulness, factual correctness)
4. **Run Evaluation**: Use `RunConfig(timeout=360)` to avoid timeouts
5. **Compare Results**: Analyze metric scores across strategies, consider latency and cost
6. **Iterate**: Tune parameters based on results, re-evaluate

## Architecture Insights

### Design Principles

1. **Consistency**: All retrieval strategies use identical LCEL chain pattern for fair comparison
2. **Modularity**: Each session is self-contained with no cross-dependencies
3. **Traceability**: LangSmith integration provides full observability into every operation
4. **Educational Focus**: Code includes extensive documentation, examples, and discussion questions
5. **Production Patterns**: Implementations follow LangChain best practices for real-world use

### Common Patterns

**LCEL Chain Construction**:
```python
chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

**LangGraph State Management**:
```python
def retrieve(state: State) -> dict:
    return {"context": retriever.invoke(state["question"])}

def generate(state: State) -> dict:
    return {"response": llm.invoke(format_prompt(state))}

graph = StateGraph(State).add_sequence([retrieve, generate])
```

**Two-Stage Retrieval**:
```python
# Over-retrieve then compress
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)
```

### Modularity

The codebase is organized for maximum learning value:

- **Independent Sessions**: Each session script is self-contained, no inter-dependencies
- **Reusable Patterns**: LCEL chains, state management, evaluation workflows are consistent
- **Flexible Configuration**: Easy to swap models, adjust parameters, experiment with strategies
- **Clear Separation**: Data loading, retrieval, generation, evaluation are distinct phases

## Troubleshooting

### Common Issues

**Rate Limiting Errors from OpenAI**:
- Add delays: `time.sleep(2)` between calls
- Use async with rate limiting for parallel operations
- Monitor usage in OpenAI dashboard

**Qdrant Collection Already Exists**:
- Use unique collection names with UUID
- Delete existing: `client.delete_collection("collection_name")`
- Use different location (`:memory:` vs disk path)

**RAGAS Evaluation Timeout**:
- Increase timeout: `RunConfig(timeout=360)` or higher
- Reduce testset size for testing
- Check API quotas and limits

**Memory Errors with Large Datasets**:
- Process documents in batches
- Use disk-based Qdrant instead of `:memory:`
- Reduce chunk sizes or document count
- Clear vector stores between runs

**LangSmith Traces Not Appearing**:
- Ensure `LANGSMITH_TRACING="true"` is set
- Verify `LANGSMITH_API_KEY` is correct
- Check project name is valid
- Allow 10-30 seconds for traces to appear

**Complete Guide**: See [API Reference](docs/04_api_reference.md#troubleshooting)

## Documentation Index

### Component Documentation

- **[01_component_inventory.md](docs/01_component_inventory.md)** - Complete catalog of all modules, classes, and functions with file references and line numbers. Includes entry points, dependencies, and module classifications.

### Architecture Documentation

- **[02_architecture_diagrams.md](diagrams/02_architecture_diagrams.md)** - Visual diagrams showing:
  - System layered architecture (Presentation → Application → Business Logic → Data Access → External Services)
  - Component relationships and module interactions
  - Retrieval strategy components and data flows
  - Class hierarchies (State, TypedDict classes, RAGAS classes)
  - Module dependency graph
  - External API and service integrations

### Flow Documentation

- **[03_data_flows.md](docs/03_data_flows.md)** - Detailed sequence diagrams for:
  - Synthetic data generation (PDF → KG → test synthesis)
  - RAG evaluation workflows (baseline and reranked)
  - All seven retrieval strategies (Naive, BM25, Compression, Multi-Query, Parent Document, Ensemble, Semantic)
  - LangSmith integration and tracing
  - State management and transitions
  - Performance considerations and optimization

### API Documentation

- **[04_api_reference.md](docs/04_api_reference.md)** - Complete API reference with:
  - Configuration options and environment variables
  - Function signatures with parameters and examples
  - Class definitions (State, AdjustedState, RAGAS classes)
  - Usage patterns and LCEL chain construction
  - Best practices for strategy selection
  - Performance considerations and trade-offs
  - Troubleshooting guide

## Project Context

### Educational Purpose

This project is designed for **learning advanced retrieval strategies for RAG systems**. It provides:

- **Hands-on experience** with seven different retrieval approaches
- **Comparative analysis** framework for understanding trade-offs
- **Production-ready patterns** using LangChain Expression Language
- **Evaluation methodologies** with RAGAS and LangSmith
- **Best practices** for RAG system design and implementation

The tutorial structure progresses from foundational concepts (synthetic data generation) through evaluation frameworks to advanced retrieval strategies, providing a complete learning journey.

### Session Progression

The three sessions build on each other:

1. **Session 07** establishes the foundation: How to create high-quality test data and evaluate RAG systems using LangSmith
2. **Session 08** introduces RAGAS metrics and LangGraph workflows, comparing baseline vs. improved retrieval
3. **Session 09** explores seven advanced strategies, demonstrating when and how to use each approach

Each session can be studied independently, but together they provide comprehensive understanding of modern RAG systems.

### Learning Outcomes

After completing this project, developers should be able to:

- **Implement** seven different retrieval strategies using LangChain
- **Evaluate** RAG systems objectively using RAGAS metrics
- **Generate** synthetic test datasets with diverse query types
- **Compare** retrieval approaches based on latency, cost, and quality
- **Choose** appropriate strategies for specific use cases
- **Build** production-ready RAG systems with proper observability
- **Debug** RAG issues using LangSmith tracing
- **Optimize** performance through strategy selection and configuration

## Additional Resources

### Related Documentation

- [LangChain Documentation](https://python.langchain.com/) - Core framework documentation
- [RAGAS Documentation](https://docs.ragas.io/) - RAG evaluation framework
- [LangSmith Documentation](https://docs.smith.langchain.com/) - Observability platform
- [OpenAI API Documentation](https://platform.openai.com/docs/) - LLM and embeddings
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking) - Reranking models
- [Qdrant Documentation](https://qdrant.tech/documentation/) - Vector database

### Further Reading

- [BM25 Algorithm Paper](https://www.nowpublishers.com/article/Details/INR-019) - Understanding sparse retrieval
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Ensemble retrieval theory
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/) - Graph-based workflows
- RAG Architecture Patterns - Best practices for production systems
- Advanced Prompting Techniques - Improving RAG quality through prompts

## Appendix

### Glossary

- **RAG**: Retrieval-Augmented Generation - Technique combining retrieval with LLM generation
- **RAGAS**: RAG Assessment - Framework for evaluating RAG systems
- **SDG**: Synthetic Data Generation - Creating test datasets programmatically
- **LCEL**: LangChain Expression Language - Declarative chain composition
- **RRF**: Reciprocal Rank Fusion - Algorithm for merging ranked lists
- **BM25**: Best Match 25 - Sparse retrieval algorithm using TF-IDF
- **Reranking**: Two-stage retrieval that refines initial results
- **Multi-Query**: Query expansion technique generating multiple variants
- **Parent Document**: Small-to-big retrieval strategy
- **Ensemble**: Combining multiple retrieval methods
- **Semantic Chunking**: Document splitting based on semantic boundaries
- **Vector Store**: Database for storing and searching embeddings
- **Embedding**: Dense vector representation of text
- **Context Window**: Text provided to LLM for generation
- **Faithfulness**: Metric measuring grounding in retrieved context
- **Hallucination**: LLM generating information not in context

### Version Information

- **Analysis Date**: October 8, 2025
- **Framework**: Architecture analysis framework (version architecture_20251008_212436)
- **Documentation Format**: Markdown with Mermaid diagrams
- **Project Version**: AIE8 Session 09 - Advanced Retrieval
- **Python Version**: 3.9+
- **LangChain Version**: >=0.3.14
- **RAGAS Version**: 0.2.10

---

*This documentation was automatically generated by the architecture analysis framework. For questions or updates, please refer to the individual component documentation files.*
