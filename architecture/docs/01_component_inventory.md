# Component Inventory

## Overview

This project is an educational repository focused on **Advanced Retrieval Techniques** for RAG (Retrieval-Augmented Generation) systems using LangChain. The project demonstrates various retrieval strategies, RAG evaluation methods using RAGAS, and synthetic data generation with LangSmith integration.

The codebase consists of:
- **4 Python module files** (converted from Jupyter notebooks for reusability)
- **1 configuration module**
- **5 Jupyter notebooks** (for interactive learning and experimentation)

The project structure is organized around three main learning sessions:
1. Session 07: Synthetic Data Generation with RAGAS and LangSmith
2. Session 08: RAGAS RAG Evaluations
3. Session 09: Advanced Retrieval Techniques

All analyzed components are **internal implementation and educational scripts** - there is no public API intended for external consumption.

---

## Public API

### Status
**None** - This is an educational project with no public API. All modules are executable scripts and examples designed for learning purposes.

---

## Internal Implementation

### Modules

#### 1. config.py
**Purpose**: Central configuration management for project paths and settings

**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`

**Key Components**:
- **PROJECT_ROOT** (line 9): Path object for project root directory
- **DATA_DIR** (line 10): Path to data directory
- **NOTEBOOKS_DIR** (line 11): Path to notebooks directory
- **DOCS_DIR** (line 12): Path to documentation directory
- **PDF_FILES** (line 15): Pattern for PDF file paths
- **CSV_FILES** (line 16): Pattern for CSV file paths
- **DEFAULT_CHUNK_SIZE** (line 19): Default chunking size (500)
- **DEFAULT_CHUNK_OVERLAP** (line 20): Default chunk overlap (50)
- **QDRANT_COLLECTION_NAME** (line 23): QDrant collection name
- **QDRANT_LOCATION** (line 24): QDrant location setting (":memory:")

**Entry Point**: Contains `if __name__ == "__main__"` block (lines 27-31) for configuration verification

---

#### 2. session07-sdg-ragas-langsmith.py
**Purpose**: Demonstrates Synthetic Data Generation using RAGAS and RAG evaluation with LangSmith

**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session07-sdg-ragas-langsmith.py`

**Key Functionality**:
- Synthetic test set generation using RAGAS knowledge graph approach
- Two RAG chain implementations (baseline and "dope")
- LangSmith evaluation with custom criteria
- Document loading from PDF files

**Dependencies**:
- LangChain (prompts, text splitters, document loaders, vectorstores)
- RAGAS (embeddings, LLMs, testset generation, knowledge graphs)
- LangSmith (evaluation, Client)
- OpenAI (ChatOpenAI, OpenAIEmbeddings)
- Qdrant (vector storage)

**Key Variables**:
- **BASELINE_PROMPT** (lines 52-59): Standard RAG prompt template
- **DOPE_PROMPT** (lines 61-70): Creative RAG prompt template with personality
- **generator_llm** (line 92): RAGAS LLM wrapper for test generation
- **generator_embeddings** (line 93): RAGAS embeddings wrapper
- **llm** (line 98): ChatOpenAI instance for RAG
- **eval_llm** (line 100): ChatOpenAI instance for evaluation

**Key Operations**:
- Knowledge graph construction (lines 107-114)
- Transform application (lines 117-120)
- Golden testset generation (lines 137-138)
- Baseline RAG chain (lines 168-171)
- Dope RAG chain (lines 202-205)
- LangSmith dataset creation (lines 221-239)
- Evaluation execution (lines 282-291, 297-306)

**No Classes or Functions Defined** - Script uses inline code execution

---

#### 3. session08-ragas-rag-evals.py
**Purpose**: RAG evaluation using RAGAS metrics with LangGraph state management

**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session08-ragas-rag-evals.py`

**Key Functionality**:
- LangGraph-based RAG implementation with state management
- Baseline retriever vs. Reranked retriever comparison
- Comprehensive RAGAS metric evaluation
- Document chunking and vector storage

**Dependencies**:
- LangChain (prompts, text splitters, document loaders, retrievers)
- LangGraph (StateGraph, START)
- RAGAS (evaluation, metrics, testset generation)
- Cohere (reranking)
- Qdrant (vector storage)

**Key Variables**:
- **BASELINE_PROMPT** (lines 62-70): RAG prompt template

**Key Functions**:

##### retrieve(state) - Line 145
**Purpose**: Retrieval node for baseline graph
- **Parameters**: state (graph state dict)
- **Returns**: dict with "context" key containing retrieved documents
- **Implementation**: Invokes retriever with question from state

##### generate(state) - Line 150
**Purpose**: Generation node for both graphs
- **Parameters**: state (graph state dict)
- **Returns**: dict with "response" key containing LLM response
- **Implementation**: Formats context and question into prompt, invokes LLM

##### retrieve_reranked(state) - Line 195
**Purpose**: Retrieval node with Cohere reranking
- **Parameters**: state (graph state dict)
- **Returns**: dict with "context" key containing reranked documents
- **Implementation**: Creates ContextualCompressionRetriever with Cohere reranker

**Key Classes**:

##### State - Lines 157-160
**Type**: TypedDict
**Purpose**: Type definition for baseline graph state
- **question**: str - User query
- **context**: List[Document] - Retrieved documents
- **response**: str - Generated response

##### AdjustedState - Lines 204-207
**Type**: TypedDict
**Purpose**: Type definition for rerank graph state (identical to State)
- **question**: str - User query
- **context**: List[Document] - Retrieved documents
- **response**: str - Generated response

**Key Operations**:
- Document splitting (lines 118-120)
- Baseline vector store creation (lines 126-140)
- Rerank vector store creation (lines 176-190)
- Baseline graph compilation (lines 163-165)
- Rerank graph compilation (lines 209-211)
- Evaluation dataset generation (lines 223-232, 235-245)
- RAGAS evaluation (lines 251-264)

---

#### 4. session09-adv-retrieval.py
**Purpose**: Consolidated implementation of advanced retrieval strategies

**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session09-adv-retrieval.py`

**Key Functionality**:
- Multiple retrieval strategies: Naive, BM25, Contextual Compression, Multi-Query, Parent Document, Ensemble, Semantic
- Vector store management with Qdrant
- RAG chain construction with LCEL
- Environment setup for OpenAI, Cohere, and LangChain APIs

**Dependencies**:
- LangChain (retrievers, storage, text splitters, prompts, runnables)
- LangChain Community (document loaders, retrievers, vectorstores)
- Cohere (reranking)
- Qdrant (vector storage)
- OpenAI (ChatOpenAI, OpenAIEmbeddings)

**Key Variables**:
- **chat_model** (line 50): ChatOpenAI("gpt-4.1-nano")
- **embeddings** (line 51): OpenAIEmbeddings("text-embedding-3-small")
- **RAG_TEMPLATE** (lines 54-64): Standard RAG prompt template
- **synthetic_usecase_data** (line 83): Loaded CSV documents

**Vector Stores and Retrievers**:
- **naive_retriever** (line 106): Basic similarity search (k=10)
- **semantic_retriever** (line 129): Semantic chunking-based retrieval (k=10)
- **parent_document_retriever** (lines 153-160): Small-to-big retrieval strategy
- **bm25_retriever** (line 166): BM25 sparse retrieval
- **compression_retriever** (lines 169-172): Cohere reranking on naive retriever
- **multi_query_retriever** (lines 175-177): Multi-query expansion
- **ensemble_retriever** (lines 180-185): Combined retrieval with equal weighting

**RAG Chains** (all using LCEL pattern):
- **naive_retrieval_chain** (lines 194-198)
- **bm25_retrieval_chain** (lines 204-208)
- **contextual_compression_retrieval_chain** (lines 214-218)
- **multi_query_retrieval_chain** (lines 224-228)
- **parent_document_retrieval_chain** (lines 234-238)
- **ensemble_retrieval_chain** (lines 244-248)
- **semantic_retrieval_chain** (lines 254-258)

**Key Operations**:
- Environment variable setup (lines 43-47)
- CSV data loading (lines 69-86)
- Naive vector store creation (lines 98-103)
- Semantic chunking (lines 112-118)
- Semantic vector store creation (lines 121-126)
- Parent document vector store setup (lines 139-148)
- Sample invocations (lines 264-324)

**No Functions or Classes Defined** - Script uses inline code and chain composition

---

#### 5. adv-retrieval.py
**Purpose**: Extended tutorial notebook covering advanced retrieval concepts with detailed explanations

**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval.py`

**Note**: This is the full educational version with markdown cells and extensive documentation, converted from a Jupyter notebook.

**Key Functionality**: Same as session09-adv-retrieval.py but with additional:
- Detailed markdown documentation and explanations
- Step-by-step tutorial structure
- Breakout room activities
- Discussion questions
- Task-based learning progression (Tasks 1-10)

**Key Sections**:
- Task 1: Dependencies setup (lines 32-46)
- Task 2: Data collection and preparation (lines 49-85)
- Task 3: QDrant setup (lines 88-107)
- Task 4: Naive RAG chain (lines 110-197)
- Task 5: BM25 retriever (lines 200-246)
- Task 6: Contextual compression/reranking (lines 248-299)
- Task 7: Multi-query retriever (lines 302-347)
- Task 8: Parent document retriever (lines 350-438)
- Task 9: Ensemble retriever (lines 441-479)
- Task 10: Semantic chunking (lines 482-562)
- Breakout Room Part #2: RAGAS evaluation activity (lines 565-595)

**Identical Implementation** to session09-adv-retrieval.py with educational scaffolding

---

### Utility Functions

**None defined** - All code is implemented as inline scripts or notebook cells

---

### Helper Classes

**None defined** - All classes are TypedDict definitions for LangGraph state management

---

### Data Files

#### Projects_with_Domains.csv
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/data/Projects_with_Domains.csv`

**Purpose**: Structured project data with domains for RAG experimentation

**Metadata Columns**:
- Project Title
- Project Domain
- Secondary Domain
- Description
- Judge Comments
- Score
- Project Name
- Judge Score

#### howpeopleuseai.pdf
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/data/howpeopleuseai.pdf`

**Purpose**: PDF document source for document loading demonstrations

---

## Entry Points

### 1. Configuration Verification
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py:27-31`

**Entry Point Type**: Command-line executable

**Usage**:
```bash
python config.py
```

**Purpose**: Prints configuration values for verification

---

### 2. Session 07 - Synthetic Data Generation
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session07-sdg-ragas-langsmith.py`

**Entry Point Type**: Interactive script (requires API keys via getpass)

**Purpose**:
- Generate synthetic test data using RAGAS
- Evaluate RAG chains with LangSmith
- Compare baseline vs. creative prompting

**Prerequisites**:
- OPENAI_API_KEY
- LANGCHAIN_API_KEY (via getpass prompts)

---

### 3. Session 08 - RAGAS Evaluations
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session08-ragas-rag-evals.py`

**Entry Point Type**: Interactive script (requires API keys via getpass)

**Purpose**:
- Compare baseline vs. reranked retrieval
- Evaluate with RAGAS metrics
- Generate evaluation datasets

**Prerequisites**:
- OPENAI_API_KEY
- COHERE_API_KEY
- LANGSMITH_API_KEY (via getpass prompts)

---

### 4. Session 09 - Advanced Retrieval
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/session09-adv-retrieval.py`

**Entry Point Type**: Interactive script (requires API keys via getpass)

**Purpose**:
- Demonstrate 7 retrieval strategies
- Build and test RAG chains
- Compare retrieval performance

**Prerequisites**:
- OPENAI_API_KEY
- COHERE_API_KEY
- LANGCHAIN_API_KEY (via getpass prompts)

---

### 5. Advanced Retrieval Tutorial
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval.py`

**Entry Point Type**: Interactive script (requires API keys via getpass)

**Purpose**:
- Full tutorial with explanations
- Step-by-step retrieval strategy implementation
- Educational activities and discussion questions

**Prerequisites**:
- OPENAI_API_KEY
- COHERE_API_KEY (via getpass prompts)

---

### Jupyter Notebooks (Entry Points)

#### 1. session07-sdg-ragas-langsmith.ipynb
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session07-sdg-ragas-langsmith.ipynb`

**Purpose**: Interactive notebook version of session 07

---

#### 2. session08-ragas-rag-evals.ipynb
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session08-ragas-rag-evals.ipynb`

**Purpose**: Interactive notebook version of session 08

---

#### 3. session09-adv-retrieval.ipynb
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session09-adv-retrieval.ipynb`

**Purpose**: Interactive notebook version of session 09

---

#### 4. Advanced_Retrieval_with_LangChain_Assignment.ipynb
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/Advanced_Retrieval_with_LangChain_Assignment.ipynb`

**Purpose**: Assignment notebook for learning assessment

---

#### 5. adv-retrieval.ipynb (temp)
**Location**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/temp/adv-retrieval.ipynb`

**Purpose**: Temporary/working version of advanced retrieval notebook

---

## Module Dependencies

### Core Dependencies

```
config.py
    └── (no internal dependencies - standalone configuration)

session07-sdg-ragas-langsmith.py
    └── External only: LangChain, RAGAS, LangSmith, OpenAI, Qdrant

session08-ragas-rag-evals.py
    └── External only: LangChain, LangGraph, RAGAS, Cohere, OpenAI, Qdrant

session09-adv-retrieval.py
    └── External only: LangChain, Cohere, OpenAI, Qdrant

adv-retrieval.py
    └── External only: LangChain, Cohere, OpenAI, Qdrant
```

### External Dependencies (from project documentation)

**Key Libraries**:
- **langchain** (>=0.3.14): Core RAG framework
- **langchain-openai** (>=0.3.33): OpenAI integration
- **langchain-community** (>=0.3.29): Community integrations
- **langchain-cohere** (==0.4.4): Cohere reranking
- **langchain-qdrant** (>=0.2.1): Qdrant vector store
- **langgraph** (==0.6.7): Graph-based workflows
- **ragas** (==0.2.10): RAG evaluation
- **cohere** (>=5.12.0,<5.13.0): Reranking service
- **qdrant-client** (>=1.7.0): Vector database client
- **pymupdf** (>=1.24.0): PDF processing

### Dependency Patterns

1. **All session scripts are independent** - No cross-module imports
2. **All scripts share similar external dependencies** - LangChain ecosystem
3. **No internal module reuse** - Each script is self-contained
4. **config.py is unused** - Not imported by any session script
5. **Data files are shared** - All scripts can access data/ directory

### API Integration Points

1. **OpenAI API**:
   - Models: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
   - Embeddings: text-embedding-3-small, text-embedding-3-large

2. **Cohere API**:
   - Reranking: rerank-v3.5

3. **LangSmith API**:
   - Tracing and evaluation
   - Dataset management

4. **Qdrant**:
   - In-memory vector storage
   - Collections: "Synthetic_Usecases", "Use Case RAG", "full_documents"

---

## Component Classification Summary

### By Type
- **Configuration Modules**: 1 (config.py)
- **Executable Scripts**: 4 (session07, session08, session09, adv-retrieval)
- **Jupyter Notebooks**: 5
- **Data Files**: 2 (CSV, PDF)

### By Purpose
- **Educational/Tutorial**: All components
- **Production/API**: None
- **Testing/Evaluation**: 2 scripts (session07, session08)
- **Demonstration**: 2 scripts (session09, adv-retrieval)

### By Complexity
- **Simple Configuration**: 1 (config.py - 32 lines)
- **Medium Scripts**: 1 (session09-adv-retrieval.py - 327 lines)
- **Complex Scripts**: 3 (session07: 333 lines, session08: 303 lines, adv-retrieval: 596 lines)

---

## Notes

1. **No Public API**: This is an educational repository with no components designed for external consumption
2. **Self-Contained Scripts**: Each script is independent and complete
3. **config.py Unused**: The configuration module is not imported by any other module
4. **Notebook-to-Script Conversion**: All .py files are converted from Jupyter notebooks (indicated by # %% cell markers)
5. **API Key Management**: All scripts use getpass for secure credential input
6. **LangSmith Integration**: All scripts include LangSmith tracing for observability
7. **Consistent Patterns**: All RAG chains follow LCEL (LangChain Expression Language) pattern
8. **Educational Focus**: Code includes extensive markdown documentation, discussion questions, and learning activities

---

**Generated**: 2025-10-08
**Project**: aie8-s09-adv-retrieval
**Analysis Framework Version**: architecture_20251008_212436
