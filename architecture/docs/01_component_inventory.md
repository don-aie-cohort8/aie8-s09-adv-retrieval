# Component Inventory

## Overview

This codebase is an educational project focused on advanced retrieval techniques using LangChain. The project demonstrates various RAG (Retrieval-Augmented Generation) strategies including naive retrieval, BM25, multi-query retrieval, parent-document retrieval, contextual compression with reranking, ensemble retrieval, and semantic chunking. The main project code consists of two Python modules and three Jupyter notebooks for interactive learning and experimentation.

**Project Name**: Advanced Retrieval with LangChain (13-advanced-retrieval)
**Main Focus**: Educational implementation of various retrieval strategies for RAG systems
**Structure**: Configuration module, main script (converted from notebook), and interactive Jupyter notebooks

---

## Public API

### Modules

#### 1. config.py
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`
**Purpose**: Central configuration module providing path management and default settings for the project
**Type**: Public configuration API
**Lines**: 1-32

#### 2. adv-retrieval.py
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval.py`
**Purpose**: Main script demonstrating all retrieval strategies (converted from notebook format)
**Type**: Public implementation/example code
**Lines**: 1-595

### Classes

No formal class definitions are present in the main project code. The codebase relies on classes from external libraries (LangChain, Cohere, OpenAI, Qdrant) but does not define custom classes.

### Functions

No standalone public functions are defined. The codebase is primarily composed of inline script code demonstrating various retrieval patterns through direct instantiation and usage of library classes.

---

## Internal Implementation

### Modules

#### config.py - Internal Constants and Paths

**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`

**Line 9**: `PROJECT_ROOT` - Path to project root directory
**Line 10**: `DATA_DIR` - Path to data directory
**Line 11**: `NOTEBOOKS_DIR` - Path to notebooks directory
**Line 12**: `DOCS_DIR` - Path to documentation directory

**Line 15**: `PDF_FILES` - Glob pattern for PDF files
**Line 16**: `CSV_FILES` - Glob pattern for CSV files

**Line 19**: `DEFAULT_CHUNK_SIZE = 500` - Default text chunk size
**Line 20**: `DEFAULT_CHUNK_OVERLAP = 50` - Default chunk overlap

**Line 23**: `QDRANT_COLLECTION_NAME = "Use Case RAG"` - Default Qdrant collection name
**Line 24**: `QDRANT_LOCATION = ":memory:"` - Default Qdrant storage location

**Lines 27-31**: Main block for configuration verification (runs when module executed directly)

### Classes

No internal classes defined in the main project code.

### Functions

No internal functions defined in the main project code.

---

## Implementation Patterns

Since the codebase is primarily educational and notebook-based, it follows a pattern-based implementation rather than traditional OOP:

### Retrieval Strategy Implementations (adv-retrieval.py)

#### 1. Naive Retrieval Pattern
**Lines**: 122-189
**Purpose**: Basic vector similarity retrieval using Qdrant
**Components**:
- Line 122: `naive_retriever` - Basic vectorstore retriever with k=10
- Lines 168-180: `naive_retrieval_chain` - LCEL chain combining retriever, prompt, and LLM
- Lines 188-194: Example queries demonstrating naive retrieval

#### 2. BM25 Retrieval Pattern
**Lines**: 200-237
**Purpose**: Sparse retrieval using Best-Matching 25 algorithm
**Components**:
- Line 212: `bm25_retriever` - BM25Retriever initialized from documents
- Lines 218-222: `bm25_retrieval_chain` - LCEL chain with BM25 retriever
- Lines 228-234: Example queries demonstrating BM25

#### 3. Contextual Compression (Reranking) Pattern
**Lines**: 248-299
**Purpose**: Document compression using Cohere's reranking model
**Components**:
- Line 274: `compressor` - CohereRerank model (rerank-v3.5)
- Lines 275-277: `compression_retriever` - ContextualCompressionRetriever wrapper
- Lines 283-287: `contextual_compression_retrieval_chain` - LCEL chain with compression
- Lines 290-296: Example queries demonstrating reranking

#### 4. Multi-Query Retrieval Pattern
**Lines**: 302-347
**Purpose**: Query expansion using LLM to generate multiple query variants
**Components**:
- Lines 321-323: `multi_query_retriever` - MultiQueryRetriever with LLM-based query expansion
- Lines 326-330: `multi_query_retrieval_chain` - LCEL chain with multi-query retriever
- Lines 333-339: Example queries demonstrating multi-query approach

#### 5. Parent Document Retrieval Pattern
**Lines**: 350-438
**Purpose**: Small-to-big strategy - search small chunks, return parent documents
**Components**:
- Line 375: `parent_docs` - Full documents designated as parents
- Line 376: `child_splitter` - RecursiveCharacterTextSplitter for creating child chunks
- Lines 386-395: `parent_document_vectorstore` - Qdrant vectorstore for child chunks
- Line 401: `store` - InMemoryStore for parent documents
- Lines 403-407: `parent_document_retriever` - ParentDocumentRetriever configuration
- Line 413: Adding documents to retriever
- Lines 419-423: `parent_document_retrieval_chain` - LCEL chain with parent-document retriever
- Lines 429-435: Example queries demonstrating parent-document approach

#### 6. Ensemble Retrieval Pattern
**Lines**: 441-479
**Purpose**: Combine multiple retrievers using Reciprocal Rank Fusion
**Components**:
- Line 452: `retriever_list` - List of all retrievers (BM25, naive, parent-document, compression, multi-query)
- Line 453: `equal_weighting` - Equal weights for all retrievers
- Lines 455-457: `ensemble_retriever` - EnsembleRetriever combining all strategies
- Lines 463-467: `ensemble_retrieval_chain` - LCEL chain with ensemble retriever
- Lines 473-479: Example queries demonstrating ensemble approach

#### 7. Semantic Chunking Pattern
**Lines**: 482-562
**Purpose**: Intelligent chunking based on semantic similarity between sentences
**Components**:
- Lines 506-509: `semantic_chunker` - SemanticChunker with percentile thresholding
- Line 515: `semantic_documents` - Documents split by semantic boundaries
- Lines 521-526: `semantic_vectorstore` - Qdrant vectorstore with semantic chunks
- Line 532: `semantic_retriever` - Retriever using semantic chunks
- Lines 538-542: `semantic_retrieval_chain` - LCEL chain with semantic chunking
- Lines 548-554: Example queries demonstrating semantic chunking

### Core Infrastructure Components (adv-retrieval.py)

#### Data Loading
**Lines**: 59-85
**Purpose**: Load and prepare CSV data with metadata
**Components**:
- Lines 62-74: `loader` - CSVLoader configuration with metadata columns
- Line 76: `synthetic_usecase_data` - Loaded documents
- Lines 78-79: Document content preparation

#### Embedding Model
**Line**: 100
**Purpose**: Initialize OpenAI embedding model
**Component**: `embeddings` - OpenAIEmbeddings(model="text-embedding-3-small")

#### Vector Store
**Lines**: 102-107
**Purpose**: Initialize primary Qdrant vector store
**Component**: `vectorstore` - Qdrant in-memory instance with synthetic usecase data

#### RAG Prompt Template
**Lines**: 130-144
**Purpose**: Standard prompt template for all RAG chains
**Component**: `RAG_TEMPLATE` and `rag_prompt` - ChatPromptTemplate for Q&A

#### Language Model
**Lines**: 152-154
**Purpose**: Initialize GPT-4 model for generation
**Component**: `chat_model` - ChatOpenAI(model="gpt-4.1-nano")

---

## Entry Points

### 1. Main Script Execution
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval.py`
**Type**: Interactive notebook-style script
**Purpose**: Educational demonstration of retrieval strategies
**Entry**: Lines 40-46 - API key setup (requires user interaction via getpass)
**Execution Flow**:
1. Lines 40-46: Environment setup (API keys)
2. Lines 59-85: Data loading and preparation
3. Lines 97-107: Vector store initialization
4. Lines 122-562: Sequential demonstration of retrieval strategies
5. Lines 568-595: Placeholder for Ragas evaluation activity

### 2. Configuration Module Verification
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`
**Type**: Standalone verification script
**Entry**: Lines 27-31 - Main block
**Purpose**: Verify configuration paths when run directly
**Execution**: `python config.py`

### 3. Jupyter Notebooks

#### session09-adv-retrieval.ipynb
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session09-adv-retrieval.ipynb`
**Type**: Interactive Jupyter notebook
**Purpose**: Hands-on learning of advanced retrieval techniques
**Cells**: 63 total (48 code, 15 markdown)
**Entry**: Cell-by-cell execution in Jupyter environment

#### session07-sdg-ragas-langsmith.ipynb
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session07-sdg-ragas-langsmith.ipynb`
**Type**: Interactive Jupyter notebook
**Purpose**: Synthetic data generation with Ragas and LangSmith integration
**Entry**: Cell-by-cell execution in Jupyter environment

#### session08-ragas-rag-evals.ipynb
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session08-ragas-rag-evals.ipynb`
**Type**: Interactive Jupyter notebook
**Purpose**: RAG evaluation using Ragas metrics
**Entry**: Cell-by-cell execution in Jupyter environment

---

## Dependencies

### Core LangChain Ecosystem
**Package**: `langchain>=0.3.19`
**Purpose**: Core framework for building LLM applications and RAG chains
**Used in**: All retrieval chain implementations (lines 59, 97, 130, 164, 210, 271, 319, 370, 450, 504)

**Package**: `langchain-experimental>=0.3.4`
**Purpose**: Experimental features including SemanticChunker
**Used in**: Lines 504-509 (semantic chunking implementation)

**Package**: `langchain-openai>=0.3.7`
**Purpose**: OpenAI integration for embeddings and LLM
**Used in**: Lines 98, 100, 152-154, 394

**Package**: `langchain-cohere==0.4.4`
**Purpose**: Cohere reranking model integration
**Used in**: Lines 272-274 (contextual compression with reranking)

**Package**: `langchain-qdrant>=0.2.0`
**Purpose**: Qdrant vector store integration
**Used in**: Lines 384-395 (parent document retriever setup)

### Vector Store & Search
**Package**: `qdrant-client>=1.13.2`
**Purpose**: Client for Qdrant vector database
**Used in**: Lines 373, 386-391 (vector store configuration)

**Package**: `rank-bm25>=0.2.2`
**Purpose**: BM25 ranking algorithm implementation
**Used in**: Lines 210-212 (BM25 retriever)

### LLM Providers
**Package**: `cohere>=5.12.0,<5.13.0`
**Purpose**: Cohere API for reranking
**Used in**: Lines 46 (API key), 274 (rerank model)

### Evaluation & Monitoring
**Package**: `ragas==0.2.10`
**Purpose**: RAG evaluation metrics and synthetic data generation
**Used in**: Lines 568-595 (evaluation activity placeholder), notebooks for evaluation

**Package**: `langgraph==0.6.7`
**Purpose**: Graph-based orchestration for complex LLM workflows
**Used in**: Potentially in evaluation and orchestration patterns

### Document Processing
**Package**: `pymupdf>=1.26.3`
**Purpose**: PDF document processing
**Used in**: Data preparation for PDF documents in data directory

**Package**: `rapidfuzz>=3.14.1`
**Purpose**: Fast fuzzy string matching
**Used in**: Potentially for document deduplication and similarity

### Development Tools
**Package**: `jupyter>=1.1.1`
**Purpose**: Jupyter notebook environment for interactive development
**Used in**: All three session notebooks

**Package**: `claude-agent-sdk>=0.1.1`
**Purpose**: Claude AI agent development SDK
**Used in**: Analysis and orchestration framework (excluded from this inventory per requirements)

---

## Data Assets

### CSV Data
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/data/Projects_with_Domains.csv`
**Purpose**: Structured project data with domains, descriptions, and judge scores
**Used in**: Lines 62-76 (loaded via CSVLoader)
**Metadata Columns**:
- Project Title
- Project Domain
- Secondary Domain
- Description
- Judge Comments
- Score
- Project Name
- Judge Score

### PDF Data
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/data/howpeopleuseai.pdf`
**Purpose**: PDF document for additional context (not directly referenced in main script)

### Evaluation Datasets (Generated)
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/baseline_evaluation_dataset.csv`
- `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/rerank_evaluation_dataset.csv`

**Purpose**: Generated evaluation datasets for comparing retrieval strategies

### Knowledge Graph Data
**File**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/usecase_data_kg.json`
**Size**: ~84 MB
**Purpose**: Knowledge graph representation of usecase data

---

## API Keys and Environment Variables

### Required Environment Variables
**Line 43**: `OPENAI_API_KEY` - OpenAI API authentication
**Line 46**: `COHERE_API_KEY` - Cohere API authentication

Both are set via `getpass.getpass()` for secure input.

---

## Key Design Patterns

### 1. LCEL (LangChain Expression Language) Chain Pattern
**Occurrences**: Lines 168-180, 218-222, 283-287, 326-330, 419-423, 463-467, 538-542
**Pattern**:
```python
chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```
**Purpose**: Declarative chain composition for RAG pipelines

### 2. Retriever Composition Pattern
**Occurrences**: Lines 275-277 (compression), 321-323 (multi-query), 455-457 (ensemble)
**Pattern**: Wrapping base retrievers with enhanced capabilities
**Purpose**: Modular enhancement of retrieval strategies

### 3. Document Metadata Preservation Pattern
**Lines**: 64-73, 78-79
**Pattern**: Extracting metadata from CSV and preserving it in document objects
**Purpose**: Enable metadata-aware retrieval and filtering

---

## Testing and Evaluation Strategy

### Evaluation Framework (Placeholder)
**Lines**: 568-595
**Purpose**: Activity section for Ragas-based evaluation
**Requirements**:
1. Create golden dataset using synthetic data generation
2. Evaluate each retriever with Ragas metrics
3. Compare performance considering cost, latency, and quality
4. Compile analysis report

**Tools**: LangSmith for latency and cost tracking (mentioned in line 590)

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Python Modules | 2 |
| Jupyter Notebooks | 3 |
| Total Lines of Code | 626 |
| Retrieval Strategies | 7 |
| LCEL Chains | 7 |
| External Dependencies | 14 |
| Data Files | 2 CSV, 1 PDF, 1 JSON |
| Configuration Constants | 8 |
| Entry Points | 5 |

---

## Architecture Notes

1. **No Custom Classes**: The codebase intentionally avoids custom class definitions, focusing instead on composing LangChain primitives
2. **Educational Focus**: Code is structured for learning, with extensive inline documentation and progressive complexity
3. **Notebook-First Design**: Primary implementation is in notebook format (adv-retrieval.py is a converted copy)
4. **Stateless Chains**: All RAG chains are stateless and can be invoked independently
5. **In-Memory Vector Stores**: All vector stores use `:memory:` location for ephemeral storage
6. **Pattern Consistency**: All retrieval strategies follow the same LCEL chain pattern for consistency
7. **Minimal Abstraction**: Direct use of library components without additional abstraction layers
