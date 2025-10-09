# Repository Architecture Documentation

## Overview

This is an **educational project** focused on exploring and comparing advanced retrieval strategies for Retrieval-Augmented Generation (RAG) systems using LangChain. The project implements 7 distinct retrieval strategies, each demonstrating different approaches to finding relevant context for large language model (LLM) responses.

**What it does**: Provides a comprehensive comparison framework for evaluating how different retrieval methods perform across various query types, enabling data scientists and ML engineers to make informed decisions about which strategy to use in production.

**Why it exists**: Understanding retrieval is critical for RAG system performance. This project serves as both a learning resource and a testing ground for comparing retrieval approaches on real data.

**Key architectural principles**:
- **Modularity**: Each retrieval strategy is independently implemented and testable
- **Consistency**: All strategies use identical LLM, prompts, and evaluation frameworks for fair comparison
- **Composability**: Strategies can be combined (as demonstrated in the Ensemble retriever)
- **Observability**: LangSmith integration enables detailed tracing and cost analysis
- **Extensibility**: New retrieval strategies can be added following established patterns

**Technology Foundation**: Built entirely on LangChain's Expression Language (LCEL), leveraging Qdrant for vector storage, OpenAI for embeddings/LLM, and Cohere for reranking.

**Target Audience**:
- Data scientists building RAG systems
- ML engineers optimizing retrieval pipelines
- Researchers comparing retrieval approaches
- Students learning advanced RAG techniques

---

## Quick Start

### For New Developers

**Where to start**: Begin with the [API Reference](docs/04_api_reference.md) to understand each retrieval strategy, then explore the [Component Inventory](docs/01_component_inventory.md) to see how it's implemented.

**Orientation path**:
1. Read "Retrieval Strategies" section below to understand the 7 approaches
2. Review [Architecture Diagrams](diagrams/02_architecture_diagrams.md) to visualize system structure
3. Examine [Data Flow Analysis](docs/03_data_flows.md) for detailed execution flows
4. Try the interactive notebook: `/notebooks/session09-adv-retrieval.ipynb`

**Key files to understand**:
- `config.py` - All configuration and paths (32 lines, very simple)
- `adv-retrieval.py` - Main implementation with all 7 strategies (595 lines)
- `notebooks/session09-adv-retrieval.ipynb` - Interactive learning environment

### For Architects

**Architectural decisions**: This project follows a **pattern-based implementation** rather than traditional OOP. It uses no custom classes, relying entirely on LangChain primitives composed via LCEL.

**Key design patterns**:
- LCEL Chain Pattern (consistent across all strategies)
- Retriever Composition Pattern (wrapping base retrievers)
- Small-to-Big Strategy (Parent Document retrieval)
- Meta-Retrieval Pattern (Ensemble combining multiple retrievers)

**Where to find architectural insights**:
- [Architecture Diagrams](diagrams/02_architecture_diagrams.md) - Visual system overview, component relationships, class hierarchies
- [Component Inventory](docs/01_component_inventory.md) - No custom classes; framework-driven architecture
- [Data Flow Analysis](docs/03_data_flows.md) - 9 sequence diagrams showing execution patterns

**Critical architectural notes**:
1. **Stateless chains**: All RAG chains are stateless and independently invocable
2. **In-memory vector stores**: Uses `:memory:` location for ephemeral storage (educational focus)
3. **No abstraction layers**: Direct use of library components without custom wrappers
4. **Notebook-first design**: Primary implementation converted from Jupyter notebook

### For Users

**Quick example** - Get started with naive retrieval in 5 steps:

```python
# 1. Set up environment
import os
import getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# 2. Initialize models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

chat_model = ChatOpenAI(model="gpt-4.1-nano")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Load your data
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./data/Projects_with_Domains.csv")
documents = loader.load()

# 4. Create vector store and retriever
vectorstore = Qdrant.from_documents(
    documents, embeddings, location=":memory:", collection_name="my_collection"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 5. Build and use RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

prompt = ChatPromptTemplate.from_template(
    "Use context to answer.\n\nQ: {question}\n\nContext: {context}"
)

chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | chat_model, "context": itemgetter("context")}
)

result = chain.invoke({"question": "What is the most common project domain?"})
print(result["response"].content)
```

**See**: [API Reference - Basic Usage](docs/04_api_reference.md#basic-usage) for complete working examples.

---

## Architecture Summary

### System Design

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│         Configuration Layer                 │  config.py - paths, settings
├─────────────────────────────────────────────┤
│         Data Loading Layer                  │  CSV → Documents with metadata
├─────────────────────────────────────────────┤
│         Embedding Layer                     │  OpenAI text-embedding-3-small
├─────────────────────────────────────────────┤
│         Vector Store Layer                  │  3x Qdrant stores + InMemoryStore
├─────────────────────────────────────────────┤
│         Retrieval Strategy Layer            │  7 distinct retrieval approaches
├─────────────────────────────────────────────┤
│         RAG Chain Layer                     │  7 LCEL chains (shared LLM/prompt)
├─────────────────────────────────────────────┤
│         Generation Layer                    │  GPT-4.1-nano + RAG prompt template
├─────────────────────────────────────────────┤
│         Output Layer                        │  Response + Context documents
└─────────────────────────────────────────────┘
```

**Key design decisions**:
1. **Multiple vector stores**: Uses 3 separate Qdrant collections optimized for different strategies (naive, semantic, parent-document)
2. **Shared components**: All chains use the same LLM and prompt, isolating retrieval as the only variable for fair comparison
3. **Parallel execution**: Ensemble retriever combines results from 5 strategies using Reciprocal Rank Fusion
4. **Two-stage patterns**: Compression (retrieve → rerank) and Parent Document (search children → return parents) use multi-stage processing

**System layers and responsibilities**:
- **Configuration**: Centralized path and setting management
- **Data**: CSV loading with metadata preservation
- **Embedding**: Convert text to 1536-dim vectors
- **Storage**: Vector databases and document stores
- **Retrieval**: 7 strategies with different trade-offs
- **Chains**: LCEL orchestration of retrieval → generation
- **Generation**: LLM response synthesis
- **Output**: Structured results with response + context

See [Architecture Diagrams - System Architecture](diagrams/02_architecture_diagrams.md#system-architecture) for detailed visualization.

### Key Architectural Patterns

#### Pattern 1: LCEL Chain Composition
**Purpose**: Declarative, type-safe chain building for RAG pipelines

**How it's used**: Every retrieval strategy follows this exact pattern:
```python
chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

**Why it matters**: Ensures consistency across all 7 strategies, making comparisons fair and code maintainable.

**Source**: Used 7 times in `adv-retrieval.py` (lines 168-180, 218-222, 283-287, 326-330, 419-423, 463-467, 538-542)

#### Pattern 2: Retriever Composition
**Purpose**: Modular enhancement of base retrieval capabilities

**Examples**:
- **Compression Retriever**: Wraps naive retriever with Cohere reranker
- **Multi-Query Retriever**: Wraps naive retriever with LLM query expansion
- **Ensemble Retriever**: Combines 5 retrievers using rank fusion

**Why it matters**: Enables building complex retrieval strategies from simple primitives without tight coupling.

**Source**: See [Component Inventory - Implementation Patterns](docs/01_component_inventory.md#implementation-patterns)

#### Pattern 3: Small-to-Big Strategy
**Purpose**: Search with precision, return with context

**How it works**: Parent Document Retriever splits documents into small chunks for precise matching, but returns full parent documents for complete context.

**Architecture**:
```
Query → Embed → Search Child Chunks → Find Parent IDs → Return Parent Documents
```

**Why it matters**: Balances precision (small chunks match better) with context (LLM needs full documents).

**Source**: `adv-retrieval.py` lines 350-438, detailed in [Data Flow Analysis - Flow 5](docs/03_data_flows.md#flow-5-parent-document-retrieval-flow)

#### Pattern 4: Meta-Retrieval (Ensemble)
**Purpose**: Leverage strengths of multiple retrieval approaches

**How it works**: Runs 5 retrievers in parallel, combines rankings using Reciprocal Rank Fusion algorithm

**Formula**: `RRF(d) = Σ (weight_i / (k + rank_i(d)))`

**Why it matters**: Most robust approach across diverse query types, combining lexical (BM25) and semantic (embeddings) search.

**Source**: `adv-retrieval.py` lines 441-479, explained in [API Reference - Strategy 6](docs/04_api_reference.md#strategy-6-ensemble-retrieval)

### Technology Stack

| Layer | Technology | Version | Why Chosen |
|-------|-----------|---------|------------|
| **Framework** | LangChain | ≥0.3.19 | Industry-standard RAG framework with LCEL |
| **Embeddings** | OpenAI text-embedding-3-small | Latest | Cost-effective, 1536 dims, excellent performance |
| **LLM** | GPT-4.1-nano | Latest | Fast, lightweight model for generation |
| **Vector Store** | Qdrant | ≥1.13.2 | High-performance, easy in-memory mode for demos |
| **Reranking** | Cohere rerank-v3.5 | 5.12.0 | State-of-the-art cross-encoder reranking |
| **Sparse Retrieval** | rank-bm25 | ≥0.2.2 | Classic BM25 algorithm for keyword matching |
| **Evaluation** | Ragas | 0.2.10 | RAG-specific evaluation metrics |
| **Monitoring** | LangSmith | Latest | Tracing, cost analysis, observability |
| **Chunking** | LangChain Experimental | ≥0.3.4 | SemanticChunker for intelligent splitting |

**Dependency philosophy**: Heavy reliance on LangChain ecosystem (6 packages) to minimize custom code and leverage battle-tested implementations.

---

## Component Overview

### Project Structure

```
/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/
├── config.py                    - Configuration module (paths, defaults)
├── adv-retrieval.py            - Main script with 7 retrieval strategies
├── data/
│   ├── Projects_with_Domains.csv  - Synthetic use case data
│   └── howpeopleuseai.pdf        - Additional PDF data
├── notebooks/
│   ├── session07-sdg-ragas-langsmith.ipynb  - Synthetic data generation
│   ├── session08-ragas-rag-evals.ipynb      - RAG evaluation
│   └── session09-adv-retrieval.ipynb        - Main interactive tutorial
└── ra_output/
    └── architecture_20251008_203645/
        ├── README.md            - This file
        ├── docs/
        │   ├── 01_component_inventory.md  - Complete component catalog
        │   ├── 03_data_flows.md          - Detailed sequence diagrams
        │   └── 04_api_reference.md       - API documentation
        └── diagrams/
            └── 02_architecture_diagrams.md - System visualizations
```

### Core Components

#### Configuration Module (`config.py`)
**Purpose**: Centralized project configuration and path management

**Key features**:
- Cross-platform path handling with `pathlib`
- Default chunking settings (500 chars, 50 overlap)
- Qdrant configuration (in-memory for demos)
- Zero external dependencies

**Usage**: `from config import PROJECT_ROOT, DATA_DIR, DEFAULT_CHUNK_SIZE`

**Source**: [API Reference - Configuration Module](docs/04_api_reference.md#configuration-module-configpy)

#### Main Implementation (`adv-retrieval.py`)
**Purpose**: Demonstrates all 7 retrieval strategies with side-by-side comparison

**Key features**:
- 595 lines of educational code
- Notebook-style structure (converted from Jupyter)
- Sequential demonstration of strategies
- Placeholder for Ragas evaluation

**Entry point**: Run cells sequentially or use as a script

**Source**: [Component Inventory - Main Implementation](docs/01_component_inventory.md#2-adv-retrievalpy)

#### Interactive Notebooks
**Purpose**: Hands-on learning and experimentation

**Available notebooks**:
1. **Session 7**: Synthetic data generation with Ragas
2. **Session 8**: RAG evaluation metrics
3. **Session 9**: Advanced retrieval strategies (main tutorial)

**Source**: All under `/notebooks/` directory

### Retrieval Strategies (The Heart of the System)

This project implements 7 distinct retrieval strategies, each with unique characteristics:

| Strategy | Type | Complexity | Latency | Cost | Best Use Case |
|----------|------|------------|---------|------|---------------|
| **1. Naive Retrieval** | Dense Vector | Low | Low | $ | Baseline semantic search, simple queries |
| **2. BM25** | Sparse Keyword | Low | Very Low | Free | Exact keyword matching, technical terms |
| **3. Multi-Query** | Query Enhancement | High | High | $$ | Ambiguous queries, improving recall |
| **4. Parent Document** | Hierarchical | Medium | Medium | $ | Documents with sections, full context needed |
| **5. Compression (Rerank)** | Post-Processing | High | High | $$$ | High precision requirements, critical queries |
| **6. Ensemble** | Meta-Retrieval | Very High | Highest | $$$$ | Robust production systems, diverse queries |
| **7. Semantic Chunking** | Smart Chunking | Medium | Medium | $$ | Documents with clear semantic boundaries |

**Strategy Selection Guide**:
- **Need exact keywords?** → BM25
- **High precision critical?** → Contextual Compression
- **Queries are ambiguous?** → Multi-Query
- **Want full document context?** → Parent Document
- **Need robust performance?** → Ensemble
- **Documents have topic transitions?** → Semantic Chunking
- **Simple and fast?** → Naive Retrieval

**Detailed documentation**: See [API Reference - Retrieval Strategies](docs/04_api_reference.md#retrieval-strategies) for complete implementation details and code examples.

---

## Data Flows

### Key Flow Patterns

The system implements 9 distinct data flows, each demonstrating different aspects of RAG retrieval:

#### 1. Data Ingestion Flow
**Purpose**: Load CSV data with metadata preservation

**Path**: CSV → CSVLoader → Documents → Metadata extraction

**Key insight**: Description field becomes `page_content`, all other CSV columns preserved as metadata for filtering.

**Source**: [Data Flow Analysis - Flow 1](docs/03_data_flows.md#flow-1-data-ingestion-and-preparation-pipeline)

#### 2. Naive Retrieval Flow (Baseline)
**Purpose**: Simple cosine similarity search

**Path**: Query → Embed → Vector similarity search → Top-k documents → Context

**Performance**: Fastest, cheapest, good baseline

**Source**: [Data Flow Analysis - Flow 2](docs/03_data_flows.md#flow-2-naive-vector-retrieval-flow)

#### 3. BM25 Flow (Lexical)
**Purpose**: Keyword-based sparse retrieval

**Path**: Query → Tokenize → BM25 scoring → Ranked documents → Context

**Advantage**: No embedding needed, excellent for exact terms

**Source**: [Data Flow Analysis - Flow 3](docs/03_data_flows.md#flow-3-bm25-sparse-retrieval-flow)

#### 4. Multi-Query Flow (Recall)
**Purpose**: Query expansion for improved recall

**Path**: Query → LLM generates variations → Parallel retrievals → Deduplicate → Context

**Trade-off**: Higher cost/latency for better coverage

**Sequence diagram**:
```
User Query → LLM (3-5 variations) → [Retrieval 1, Retrieval 2, ...] → Merge → Context
```

**Source**: [Data Flow Analysis - Flow 4](docs/03_data_flows.md#flow-4-multi-query-retrieval-flow)

#### 5. Parent Document Flow (Small-to-Big)
**Purpose**: Precise matching with complete context

**Architecture**:
```
Setup Phase: Documents → Split children → Store in vector DB → Store parents in InMemoryStore
Query Phase: Query → Search children → Get parent IDs → Return full parents
```

**Why it works**: Small chunks match precisely, parent documents provide full context to LLM.

**Source**: [Data Flow Analysis - Flow 5](docs/03_data_flows.md#flow-5-parent-document-retrieval-flow)

#### 6. Compression Flow (Two-Stage)
**Purpose**: Retrieve candidates, then rerank for precision

**Stages**:
1. **Stage 1**: Naive retriever gets k=10 candidates (broad recall)
2. **Stage 2**: Cohere rerank scores query-document pairs (high precision)

**Result**: More accurate than embeddings alone due to cross-encoder scoring.

**Source**: [Data Flow Analysis - Flow 6](docs/03_data_flows.md#flow-6-contextual-compression-reranking-flow)

#### 7. Ensemble Flow (Meta-Retrieval)
**Purpose**: Combine strengths of multiple strategies

**Algorithm**: Reciprocal Rank Fusion (RRF)
```
RRF(doc) = Σ (weight_i / (60 + rank_i(doc)))
```

**Retriever combination**:
- BM25 (keywords) + Naive (semantic) + Multi-Query (recall) + Parent Document (context) + Compression (precision)

**Benefits**: Most robust across query types, reduces individual weaknesses.

**Source**: [Data Flow Analysis - Flow 7](docs/03_data_flows.md#flow-7-ensemble-retrieval-flow)

#### 8. Semantic Chunking Flow
**Purpose**: Create chunks based on semantic coherence

**Process**:
1. Split document into sentences
2. Embed each sentence
3. Calculate inter-sentence similarity
4. Identify semantic breaks (percentile threshold)
5. Group sentences into coherent chunks

**Trade-off**: Higher setup cost (embed every sentence) for better chunk quality.

**Source**: [Data Flow Analysis - Flow 8](docs/03_data_flows.md#flow-8-semantic-chunking-preparation-flow)

#### 9. Complete RAG Chain Flow
**Purpose**: End-to-end LCEL pipeline execution

**LCEL stages**:
```python
Stage 1: Build context dict     {"context": retriever(question), "question": question}
Stage 2: Passthrough context    RunnablePassthrough.assign(context=...)
Stage 3: Generate response       {"response": prompt | llm, "context": context}
```

**Output**: `{"response": AIMessage, "context": [Documents]}`

**Source**: [Data Flow Analysis - Flow 9](docs/03_data_flows.md#flow-9-complete-rag-chain-execution-flow)

### Data Pipeline Overview

High-level data movement through the system:

```
CSV Data
   ↓
Documents (with metadata)
   ↓
Embeddings (1536-dim vectors)
   ↓
Vector Stores (3x Qdrant collections)
   ↓
Retrievers (7 strategies)
   ↓
RAG Chains (LCEL)
   ↓
LLM Generation (GPT-4.1-nano)
   ↓
Response + Context
```

**Key characteristics**:
- **Single data source**: One CSV file feeds all strategies
- **Multiple vector stores**: 3 different Qdrant collections for different chunk types
- **Shared generation**: All strategies use same LLM and prompt for fair comparison
- **Transparent output**: Returns both response and source documents

See [Architecture Diagrams - Data Flow](diagrams/02_architecture_diagrams.md#data-flow) for visual representation.

---

## Getting Started

### Prerequisites

**Required**:
- Python 3.10 or higher
- API keys for:
  - OpenAI (embeddings + LLM)
  - Cohere (reranking)
  - LangSmith (optional, for tracing)

**Recommended**:
- Jupyter Lab or VS Code (for notebooks)
- 8GB+ RAM (for in-memory vector stores)
- Git (for cloning repository)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd aie8-s09-adv-retrieval

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# Verify installation
python config.py
```

**Expected output**:
```
Project Root: /home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval
Data Directory: /home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/data
Data Directory Exists: True
PDF Files: /home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/data/*.pdf
```

### Configuration

**API Keys** - Set environment variables:

```python
import os
import getpass

# Required
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")

# Optional (for tracing)
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "advanced-retrieval-demo"
```

**Or use `.env` file** (recommended):

```bash
# .env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=advanced-retrieval-demo
```

Then load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Running Your First Retrieval

**Option 1: Interactive Notebook** (recommended for learning)

```bash
jupyter lab notebooks/session09-adv-retrieval.ipynb
```

Run cells sequentially to explore each strategy.

**Option 2: Python Script**

```bash
python adv-retrieval.py
```

Enter API keys when prompted, then observe each strategy's output.

**Option 3: Quick Test**

```python
from config import DATA_DIR
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Load data
loader = CSVLoader(file_path=f"{DATA_DIR}/Projects_with_Domains.csv")
docs = loader.load()

# Initialize
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chat_model = ChatOpenAI(model="gpt-4.1-nano")

# Create vector store
vectorstore = Qdrant.from_documents(
    docs, embeddings, location=":memory:", collection_name="test"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Build chain
prompt = ChatPromptTemplate.from_template(
    "Use context to answer.\n\nQ: {question}\n\nContext: {context}"
)
chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | chat_model, "context": itemgetter("context")}
)

# Query
result = chain.invoke({"question": "What is the most common project domain?"})
print(f"Answer: {result['response'].content}")
print(f"Retrieved {len(result['context'])} documents")
```

---

## Usage Guide

### Choosing a Retrieval Strategy

**Decision tree**:

```
Start Here
    │
    ├─ Need exact keyword matching? ──YES──> BM25
    │                                   NO
    ├─ High precision critical? ──YES──> Contextual Compression (Reranking)
    │                              NO
    ├─ Query is ambiguous/multi-faceted? ──YES──> Multi-Query
    │                                       NO
    ├─ Documents have sections, need full context? ──YES──> Parent Document
    │                                                 NO
    ├─ Want robust performance across query types? ──YES──> Ensemble
    │                                                 NO
    ├─ Documents have clear topic boundaries? ──YES──> Semantic Chunking
    │                                           NO
    └─ Default/baseline ────> Naive Retrieval
```

**Strategy characteristics**:

| When... | Use | Why |
|---------|-----|-----|
| Query has specific technical terms | BM25 | Exact keyword matching outperforms semantic |
| Precision is more important than speed | Compression | Reranking filters to most relevant |
| User query might be poorly worded | Multi-Query | LLM reformulations improve recall |
| Documents are long with distinct sections | Parent Document | Search precise, return complete |
| Query types are unpredictable | Ensemble | Combines multiple approaches |
| Content has natural semantic structure | Semantic Chunking | Preserves topic coherence |
| Starting point or simple use case | Naive | Fast, cheap, good baseline |

### Common Patterns

#### Pattern 1: Hybrid Search (BM25 + Semantic)

**Purpose**: Best of both lexical and semantic worlds

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Create both retrievers
bm25 = BM25Retriever.from_documents(documents)
semantic = vectorstore.as_retriever(search_kwargs={"k": 10})

# Combine with 50/50 weighting
hybrid = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.5, 0.5]
)

# Use in chain
chain = (
    {"context": itemgetter("question") | hybrid, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

**When to use**: Production systems with diverse query types, when you can't predict whether queries will be keyword-based or semantic.

#### Pattern 2: Two-Stage Retrieval (Recall → Precision)

**Purpose**: Cast wide net, then filter to best results

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Stage 1: High recall via multi-query
base = vectorstore.as_retriever(search_kwargs={"k": 20})
recall_retriever = MultiQueryRetriever.from_llm(retriever=base, llm=chat_model)

# Stage 2: High precision via reranking
reranker = CohereRerank(model="rerank-v3.5")
precision_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=recall_retriever
)

# Use precision_retriever in chain
```

**When to use**: Research applications, when missing relevant documents is costly, when precision and recall both matter.

#### Pattern 3: Metadata Filtering

**Purpose**: Constrain retrieval to specific document subsets

```python
# Filter by project domain
filtered_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {"Project Domain": "Finance / FinTech"}
    }
)

# Use in chain for domain-specific queries
result = chain.invoke({
    "question": "What fintech projects scored highest?"
})
```

**When to use**: Multi-tenant systems, when documents have clear categories, when users know their domain.

**See**: [API Reference - Common Patterns](docs/04_api_reference.md#common-patterns) for more examples including fallback chains and routing.

---

## Performance Considerations

### Strategy Comparison

| Strategy | API Calls | Latency | Cost/Query | Precision | Recall | Production Ready? |
|----------|-----------|---------|------------|-----------|--------|-------------------|
| Naive | Low (1-2) | ~200ms | $0.001 | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Yes |
| BM25 | None | ~50ms | Free | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Yes |
| Multi-Query | High (3-4) | ~800ms | $0.003 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ Expensive |
| Parent Document | Low (1-2) | ~250ms | $0.001 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes |
| Compression | High (2-3) | ~600ms | $0.005 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⚠️ Expensive |
| Ensemble | Very High (5-10) | ~1500ms | $0.010 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ Very Expensive |
| Semantic Chunking | High (setup) | ~200ms (query) | $0.002 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes |

**Notes**:
- Costs are approximate and vary based on document/query length
- Latency measured on typical queries, can vary significantly
- Precision/Recall ratings are relative, not absolute metrics

### Optimization Tips

#### 1. Cache Embeddings
**Problem**: Re-embedding same documents wastes money and time

**Solution**:
```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace="text-embedding-3-small"
)

# Use cached_embeddings instead of embeddings
vectorstore = Qdrant.from_documents(docs, cached_embeddings, ...)
```

**Impact**: 100% cost savings on repeated embeddings, 10-50x faster document loading.

#### 2. Limit Retrieved Documents
**Problem**: Retrieving too many documents increases cost and latency

**Solution**:
```python
# Start with fewer documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Instead of 10

# For reranking, retrieve more initially but filter aggressively
base = vectorstore.as_retriever(search_kwargs={"k": 20})
compression = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base
)
```

**Impact**: 2x faster for k=5 vs k=10, proportional cost reduction.

#### 3. Use Async for Concurrency
**Problem**: Sequential processing wastes time

**Solution**:
```python
import asyncio

async def process_queries(questions):
    tasks = [chain.ainvoke({"question": q}) for q in questions]
    return await asyncio.gather(*tasks)

results = asyncio.run(process_queries([
    "Question 1", "Question 2", "Question 3"
]))
```

**Impact**: Near-linear speedup for independent queries.

#### 4. Batch Embedding
**Problem**: Individual API calls are inefficient

**Solution**:
```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=1000  # Batch size for API calls
)
```

**Impact**: Higher throughput, more efficient API usage.

**See**: [API Reference - Performance Optimization](docs/04_api_reference.md#performance-optimization) for more tips.

---

## Architecture Deep Dives

### Detailed Documentation

For comprehensive technical details, see:

#### 1. [Component Inventory](docs/01_component_inventory.md)
**What it covers**:
- Complete module and function catalog (2 Python modules, 3 notebooks)
- Implementation details for all 7 retrieval strategies
- Line-by-line source references
- Dependency analysis (14 external packages)
- Data assets and configuration

**Key insights**:
- No custom classes defined - framework-driven architecture
- 7 LCEL chains following identical pattern
- Pattern-based implementation for educational clarity
- Minimal abstraction over LangChain primitives

**When to read**: Understanding implementation details, contributing code, debugging specific components.

#### 2. [Architecture Diagrams](diagrams/02_architecture_diagrams.md)
**What it covers**:
- System architecture visualization (Mermaid diagrams)
- Component relationships and dependencies
- Class hierarchies from LangChain
- Module dependency graph
- Data flow diagrams
- Retrieval strategy taxonomy

**Key insights**:
- Layered architecture with 8 distinct layers
- 3 separate Qdrant vector stores for different strategies
- Hub-and-spoke pattern with document corpus at center
- Ensemble combines 5 retrievers using RRF

**When to read**: Getting architectural overview, understanding system design, planning extensions.

#### 3. [Data Flow Analysis](docs/03_data_flows.md)
**What it covers**:
- 9 detailed sequence diagrams showing execution flows
- Step-by-step breakdowns of each retrieval strategy
- LCEL chain execution pipeline
- Data transformation at each stage

**Key insights**:
- All flows share common entry (user query) and exit (LLM response)
- Two-stage patterns in Compression and Parent Document retrievers
- Parallel execution in Ensemble retriever
- Semantic chunking uses percentile thresholding

**When to read**: Debugging retrieval issues, understanding execution order, optimizing performance.

#### 4. [API Reference](docs/04_api_reference.md)
**What it covers**:
- Complete API documentation for all 7 strategies
- Configuration module reference
- Usage patterns and best practices
- Common patterns (hybrid search, two-stage retrieval)
- Troubleshooting guide
- Performance optimization tips

**Key insights**:
- All chains follow same LCEL pattern with different retrievers
- Strategy selection decision tree
- Cost vs performance trade-offs
- Caching and batching recommendations

**When to read**: Implementing retrieval strategies, troubleshooting errors, optimizing costs.

---

## Key Insights

### Architectural Highlights

#### 1. Framework-Driven Design Eliminates Custom Code
**Insight**: The entire system uses zero custom classes, relying on LangChain primitives composed via LCEL.

**Why it matters**:
- Lower maintenance burden
- Battle-tested implementations
- Easy to understand for LangChain users
- Faster development

**Where to learn more**: [Component Inventory - Architecture Notes](docs/01_component_inventory.md#architecture-notes)

#### 2. Consistent LCEL Pattern Enables Fair Comparison
**Insight**: All 7 retrieval strategies use identical chain structure, LLM, and prompt - only the retriever varies.

**Pattern**:
```python
{"context": itemgetter("question") | RETRIEVER, "question": itemgetter("question")}
| RunnablePassthrough.assign(context=itemgetter("context"))
| {"response": rag_prompt | chat_model, "context": itemgetter("context")}
```

**Why it matters**: Isolates retrieval as the only variable, making performance comparisons scientifically valid.

**Where to learn more**: [Component Inventory - Key Design Patterns](docs/01_component_inventory.md#key-design-patterns)

#### 3. Multiple Vector Stores Optimize for Different Strategies
**Insight**: System uses 3 separate Qdrant collections, each optimized for specific retrieval approaches.

**Collections**:
- `Synthetic_Usecases`: Standard chunks for naive retrieval
- `Synthetic_Usecase_Data_Semantic_Chunks`: Semantic chunks with variable boundaries
- `full_documents`: Child chunks for parent document retrieval

**Why it matters**: One-size-fits-all vector stores are suboptimal; different strategies need different chunking.

**Where to learn more**: [Architecture Diagrams - System Architecture](diagrams/02_architecture_diagrams.md#system-architecture)

#### 4. Ensemble Retriever Demonstrates Meta-Retrieval Pattern
**Insight**: Ensemble combines 5 retrievers (BM25, Naive, Multi-Query, Parent, Compression) using Reciprocal Rank Fusion.

**Algorithm**: `RRF(d) = Σ (weight_i / (60 + rank_i(d)))`

**Why it matters**:
- Most robust approach across query types
- Combines lexical and semantic search
- Production-ready pattern for high-stakes applications

**Trade-offs**: Highest latency and cost, but best overall performance.

**Where to learn more**: [Data Flow Analysis - Flow 7](docs/03_data_flows.md#flow-7-ensemble-retrieval-flow)

#### 5. Small-to-Big Strategy Balances Precision and Context
**Insight**: Parent Document Retriever searches small chunks (precise) but returns full documents (contextual).

**Architecture**: Child chunks in vector store, parent documents in InMemoryStore, linked by IDs.

**Why it matters**: Solves fundamental trade-off in chunk-based retrieval - small chunks match better, large chunks provide better context for LLMs.

**Where to learn more**: [API Reference - Strategy 5](docs/04_api_reference.md#strategy-5-parent-document-retrieval)

#### 6. Semantic Chunking Adapts to Content Structure
**Insight**: Instead of fixed-size chunks, splits documents at semantic boundaries using sentence embedding similarity.

**Methods**: Percentile, standard deviation, interquartile, gradient - each suited for different content types.

**Why it matters**:
- Preserves topic coherence
- No mid-topic splits
- Variable chunk sizes adapt to content

**Trade-offs**: Expensive setup (embed every sentence), better quality.

**Where to learn more**: [API Reference - Strategy 7](docs/04_api_reference.md#strategy-7-semantic-chunking)

### Design Philosophy

**Core principles evident in the codebase**:

1. **Education First**: Code is optimized for learning, not production. Extensive inline documentation, progressive complexity, notebook-first design.

2. **Composability Over Inheritance**: Uses function composition and LCEL chains rather than class hierarchies. Easier to understand and modify.

3. **Empirical Comparison**: Architecture designed for side-by-side strategy comparison with controlled variables (same data, LLM, prompt).

4. **Observability Built-In**: LangSmith integration from the start for tracing, cost analysis, debugging.

5. **Minimal Abstraction**: Direct use of library components without custom wrappers. Lower cognitive load, clearer data flow.

6. **Stateless Chains**: All RAG chains are stateless and independently invocable. Easy to test, parallelize, cache.

### Evolution and Future

**Potential extensions** (evident from architecture):

1. **Evaluation Framework**: Placeholder exists for Ragas-based evaluation (lines 568-595 in `adv-retrieval.py`). Future work will compare strategies empirically.

2. **Persistent Vector Stores**: Currently uses `:memory:` for demos. Production would use persistent Qdrant or cloud-hosted service.

3. **Custom Retrievers**: Pattern established makes adding new strategies straightforward - follow same LCEL chain structure.

4. **Metadata Filtering**: CSV columns preserved as metadata, enabling domain-specific retrieval.

5. **A/B Testing**: Architecture supports running multiple strategies in parallel for live traffic comparison.

---

## Troubleshooting

### Common Issues

#### 1. Empty or Irrelevant Results
**Symptoms**: Retriever returns no documents or clearly wrong ones

**Quick fixes**:
```python
# Verify vector store has documents
print(f"Docs in store: {len(vectorstore.similarity_search('test', k=1))}")

# Try BM25 for exact keywords
bm25 = BM25Retriever.from_documents(documents)
results = bm25.invoke("your query")

# Lower similarity threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
```

#### 2. High Latency
**Symptoms**: Queries take several seconds

**Quick fixes**:
```python
# Reduce k value
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Use async for parallel queries
results = await asyncio.gather(*[chain.ainvoke({"question": q}) for q in queries])

# Profile to find bottleneck
import time
start = time.time()
docs = retriever.invoke(query)
print(f"Retrieval: {time.time() - start:.2f}s")
```

#### 3. API Rate Limits
**Symptoms**: `RateLimitError` or 429 errors

**Quick fixes**:
```python
# Add retry logic
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def embed_with_retry(texts):
    return embeddings.embed_documents(texts)

# Batch with delays
for i in range(0, len(docs), batch_size):
    vectorstore.add_documents(docs[i:i+batch_size])
    time.sleep(1)
```

#### 4. High API Costs
**Symptoms**: Unexpected bills

**Quick fixes**:
```python
# Cache embeddings aggressively
from langchain.embeddings import CacheBackedEmbeddings
cached = CacheBackedEmbeddings.from_bytes_store(embeddings, LocalFileStore("./cache"))

# Limit ensemble components
ensemble = EnsembleRetriever(
    retrievers=[bm25, semantic],  # Just 2 instead of 5
    weights=[0.5, 0.5]
)

# Use cheaper models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Cheapest
chat_model = ChatOpenAI(model="gpt-4.1-nano")  # Cheaper than gpt-4
```

#### 5. Out of Memory Errors
**Symptoms**: Program crashes, especially with large document sets

**Quick fixes**:
```python
# Process in batches
for i in range(0, len(documents), 100):
    vectorstore.add_documents(documents[i:i+100])

# Use persistent storage
client = QdrantClient(path="./qdrant_storage")

# Enable on-disk vectors
client.create_collection(
    collection_name="large",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE, on_disk=True)
)
```

**For complete troubleshooting guide, see**: [API Reference - Troubleshooting](docs/04_api_reference.md#troubleshooting)

---

## Contributing

### Understanding the Codebase

**New contributor workflow**:

1. **Read this README** for high-level understanding
2. **Review [API Reference](docs/04_api_reference.md)** for each strategy's implementation
3. **Examine [Component Inventory](docs/01_component_inventory.md)** for code organization
4. **Study [Data Flow Analysis](docs/03_data_flows.md)** for execution patterns
5. **Run interactive notebook**: `notebooks/session09-adv-retrieval.ipynb`

**Key files for contributors**:
- `config.py` (32 lines) - Start here, very simple
- `adv-retrieval.py` (595 lines) - Main implementation
- `notebooks/session09-adv-retrieval.ipynb` - Interactive version

### Adding New Retrieval Strategies

**Pattern to follow**:

1. **Create retriever** following LangChain retriever interface
2. **Build LCEL chain** using the standard pattern:
   ```python
   new_strategy_chain = (
       {"context": itemgetter("question") | new_retriever, "question": itemgetter("question")}
       | RunnablePassthrough.assign(context=itemgetter("context"))
       | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
   )
   ```
3. **Test with sample queries** to verify retrieval quality
4. **Add to Ensemble** (optional) to compare performance
5. **Document in API Reference** following existing strategy format

**Example - Adding a new "Hybrid BM25+Semantic" strategy**:

```python
from langchain.retrievers import EnsembleRetriever

# Create hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, naive_retriever],
    weights=[0.5, 0.5]
)

# Build chain (same pattern as others)
hybrid_retrieval_chain = (
    {"context": itemgetter("question") | hybrid_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# Test
result = hybrid_retrieval_chain.invoke({"question": "Test query"})
```

**See**: [Component Inventory - Implementation Patterns](docs/01_component_inventory.md#implementation-patterns) for existing strategy implementations.

---

## References

### Internal Documentation

- **[Component Inventory](docs/01_component_inventory.md)** - Complete catalog of modules, functions, dependencies
- **[Architecture Diagrams](diagrams/02_architecture_diagrams.md)** - Visual system overview with Mermaid diagrams
- **[Data Flow Analysis](docs/03_data_flows.md)** - 9 detailed sequence diagrams
- **[API Reference](docs/04_api_reference.md)** - Complete API docs with examples

### External Resources

#### LangChain Documentation
- [LangChain Main Docs](https://python.langchain.com/) - Official documentation
- [Retrievers Guide](https://python.langchain.com/docs/modules/data_connection/retrievers/) - Retriever implementations
- [LCEL Guide](https://python.langchain.com/docs/expression_language/) - Expression Language syntax
- [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/) - Vector database integrations

#### Retrieval Algorithms
- [BM25 Algorithm](https://www.nowpublishers.com/article/Details/INR-019) - Okapi BM25 ranking function
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - RRF algorithm paper
- [Semantic Chunking](https://python.langchain.com/docs/how_to/semantic-chunker/) - LangChain implementation

#### Vector Databases
- [Qdrant Docs](https://qdrant.tech/documentation/) - Vector search engine documentation
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client) - Official Python SDK

#### APIs and Models
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) - text-embedding-3-small docs
- [OpenAI Models](https://platform.openai.com/docs/models) - GPT-4.1-nano specifications
- [Cohere Rerank](https://docs.cohere.com/docs/reranking) - rerank-v3.5 model guide

#### Observability
- [LangSmith](https://docs.smith.langchain.com/) - Tracing and monitoring platform
- [LangSmith Tracing](https://docs.smith.langchain.com/tracing) - How to trace chains

### Research Papers

- **Dense Retrieval**: [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) - DPR paper from Facebook AI
- **Reranking**: [RankGPT: Reranking with LLMs](https://arxiv.org/abs/2304.09542) - LLM-based reranking
- **Hybrid Search**: [Complementarity of Lexical and Semantic Matching](https://arxiv.org/abs/2104.08663) - Combining BM25 and embeddings

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **BM25** | Best Matching 25 - sparse retrieval algorithm based on TF-IDF |
| **Dense Retrieval** | Vector-based semantic search using embeddings |
| **Embedding** | Numerical vector representation of text (1536 dimensions for text-embedding-3-small) |
| **Ensemble Retrieval** | Combining multiple retrievers using fusion algorithms |
| **LCEL** | LangChain Expression Language - declarative chain composition syntax |
| **Naive Retrieval** | Simple cosine similarity search (baseline) |
| **RAG** | Retrieval-Augmented Generation - LLM + retrieved context |
| **Reranking** | Two-stage retrieval: retrieve candidates, then rerank for precision |
| **RRF** | Reciprocal Rank Fusion - algorithm for combining ranked lists |
| **Semantic Chunking** | Splitting text at semantic boundaries rather than fixed sizes |
| **Sparse Retrieval** | Keyword-based search using bag-of-words representations |
| **Vector Store** | Database optimized for similarity search over embeddings (e.g., Qdrant) |

### Architecture Decision Records

While no formal ADRs exist, key architectural decisions evident from the codebase:

#### ADR-001: Framework-Driven Implementation (No Custom Classes)
**Context**: Educational project needs to be understandable and maintainable

**Decision**: Use LangChain primitives directly without custom class abstractions

**Rationale**:
- Lower cognitive load for learners
- Battle-tested implementations
- Easier to upgrade dependencies
- Standard patterns familiar to LangChain users

**Consequences**:
- Less flexibility for custom behavior
- Tightly coupled to LangChain API changes
- Requires understanding LangChain internals

#### ADR-002: In-Memory Vector Stores for Demo
**Context**: Project is educational, not production-focused

**Decision**: Use `:memory:` location for Qdrant stores

**Rationale**:
- No setup required
- Faster to get started
- No cleanup needed
- Perfect for notebooks

**Consequences**:
- Data lost on restart
- Not suitable for large datasets
- Production deployments need modification

#### ADR-003: Consistent LCEL Chain Pattern
**Context**: Need to compare 7 retrieval strategies fairly

**Decision**: All chains use identical structure, LLM, and prompt - only retriever varies

**Rationale**:
- Isolates retrieval as the only variable
- Scientifically valid comparisons
- Easy to understand pattern
- Copy-paste consistency

**Consequences**:
- Less flexibility per strategy
- Prompt might not be optimal for all retrievers
- Hard to customize individual chains

#### ADR-004: Multiple Vector Stores for Different Chunking Strategies
**Context**: Different retrieval strategies need different chunk types

**Decision**: Use 3 separate Qdrant collections (standard, semantic, child chunks)

**Rationale**:
- Semantic chunking needs variable-size chunks
- Parent document retrieval needs child chunks
- One size doesn't fit all

**Consequences**:
- Higher memory usage
- More complex setup
- Better retrieval quality

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-08 | Initial architecture documentation |
| - | - | 7 retrieval strategies implemented |
| - | - | Complete API reference and diagrams |
| - | - | Educational notebooks added |

---

**Generated**: 2025-10-08
**Documentation Version**: 1.0
**Project**: Advanced Retrieval with LangChain (aie8-s09-adv-retrieval)
**Maintained By**: AIE8 Cohort

**Source Files**:
- Configuration: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`
- Main Implementation: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval.py`
- Interactive Notebook: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session09-adv-retrieval.ipynb`

**Documentation Suite**:
- This README provides high-level overview and quick start
- [Component Inventory](docs/01_component_inventory.md) for implementation details
- [Architecture Diagrams](diagrams/02_architecture_diagrams.md) for visual understanding
- [Data Flow Analysis](docs/03_data_flows.md) for execution patterns
- [API Reference](docs/04_api_reference.md) for complete API documentation

---

**Questions or Issues?** Refer to the detailed documentation linked above, or review the troubleshooting section for common problems and solutions.
