# API Reference

## Overview

This API reference documents the main project code for the Advanced Retrieval RAG system. The project implements and compares 7 different retrieval strategies for Retrieval-Augmented Generation (RAG) using LangChain.

**Target Audience**: Developers and data scientists working with RAG systems

**Project Components**:
- `config.py` - Configuration module with paths and settings
- `adv-retrieval-with-langchain.py` - Main implementation script with all retrieval strategies
- Jupyter notebooks for interactive exploration and learning

## Configuration Module (`config.py`)

### Overview

The configuration module provides centralized path management and default settings for the Advanced Retrieval project. It uses Python's `pathlib` for cross-platform path handling and defines constants for vector stores, chunking, and data sources.

**Source**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`

### Constants and Settings

#### Directory Paths

##### `PROJECT_ROOT`
- **Type**: `pathlib.Path`
- **Value**: Parent directory of the config file
- **Purpose**: Root directory for all project paths
- **Source**: `config.py:9`

##### `DATA_DIR`
- **Type**: `pathlib.Path`
- **Value**: `PROJECT_ROOT / "data"`
- **Purpose**: Directory containing CSV and PDF data files
- **Source**: `config.py:10`

##### `NOTEBOOKS_DIR`
- **Type**: `pathlib.Path`
- **Value**: `PROJECT_ROOT / "notebooks"`
- **Purpose**: Directory containing Jupyter notebooks
- **Source**: `config.py:11`

##### `DOCS_DIR`
- **Type**: `pathlib.Path`
- **Value**: `PROJECT_ROOT / "docs"`
- **Purpose**: Directory for project documentation
- **Source**: `config.py:12`

#### Data File Patterns

##### `PDF_FILES`
- **Type**: `pathlib.Path` (glob pattern)
- **Value**: `DATA_DIR / "*.pdf"`
- **Purpose**: Pattern for locating PDF files in the data directory
- **Source**: `config.py:15`

##### `CSV_FILES`
- **Type**: `pathlib.Path` (glob pattern)
- **Value**: `DATA_DIR / "*.csv"`
- **Purpose**: Pattern for locating CSV files in the data directory
- **Source**: `config.py:16`

#### Model Configuration

##### `DEFAULT_CHUNK_SIZE`
- **Type**: `int`
- **Value**: `500`
- **Purpose**: Default character count for text chunks when splitting documents
- **Source**: `config.py:19`
- **Usage**: Used with text splitters when no specific chunk size is provided

##### `DEFAULT_CHUNK_OVERLAP`
- **Type**: `int`
- **Value**: `50`
- **Purpose**: Default character overlap between consecutive chunks
- **Source**: `config.py:20`
- **Usage**: Ensures continuity between chunks to preserve context

#### Vector Store Configuration

##### `QDRANT_COLLECTION_NAME`
- **Type**: `str`
- **Value**: `"Use Case RAG"`
- **Purpose**: Default collection name for Qdrant vector store
- **Source**: `config.py:23`

##### `QDRANT_LOCATION`
- **Type**: `str`
- **Value**: `":memory:"`
- **Purpose**: Qdrant storage location (in-memory for development/testing)
- **Source**: `config.py:24`
- **Note**: For production, change to a persistent location or remote URL

### Usage Example

```python
from config import (
    PROJECT_ROOT,
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    QDRANT_COLLECTION_NAME,
    QDRANT_LOCATION
)

# Verify configuration
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Data Directory Exists: {DATA_DIR.exists()}")

# Use in document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)

# Use in vector store setup
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

vectorstore = Qdrant.from_documents(
    documents,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    location=QDRANT_LOCATION,
    collection_name=QDRANT_COLLECTION_NAME
)
```

### Verification Script

The config module includes a verification script that can be run directly:

```bash
python config.py
```

This will print:
- Project root path
- Data directory path
- Whether data directory exists
- PDF files pattern

**Source**: `config.py:27-32`

## Main Module (`adv-retrieval-with-langchain.py`)

### Overview

This is the primary implementation module that demonstrates 7 different retrieval strategies for RAG systems. It's structured as a Python script with Jupyter notebook cell markers (`# %%`) for use in VS Code or Jupyter.

**Source**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval-with-langchain.py`

**Architecture**: Script-based implementation using LangChain Expression Language (LCEL) for chain composition

### Environment Setup

The script requires three API keys:

```python
import os
import getpass

# Required API keys
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIM - SDG - {uuid4().hex[0:8]}"
```

**Source**: `adv-retrieval-with-langchain.py:43-47`

**Required Services**:
- OpenAI: For embeddings (text-embedding-3-small) and LLM (gpt-4.1-nano)
- Cohere: For reranking (rerank-v3.5)
- LangSmith: For tracing and observability (optional but recommended)

### Core Components

#### Model Initialization

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

chat_model = ChatOpenAI(model="gpt-4.1-nano")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

**Source**: `adv-retrieval-with-langchain.py:50-51`

**Models Used**:
- **LLM**: `gpt-4.1-nano` - Fast, lightweight model for generation
- **Embeddings**: `text-embedding-3-small` - 1536 dimensions, cost-effective

#### RAG Prompt Template

```python
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
```

**Source**: `adv-retrieval-with-langchain.py:54-66`

**Template Variables**:
- `{question}`: User query
- `{context}`: Retrieved documents/context

#### Data Loading

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path=f"../data/Projects_with_Domains.csv",
    metadata_columns=[
        "Project Title",
        "Project Domain",
        "Secondary Domain",
        "Description",
        "Judge Comments",
        "Score",
        "Project Name",
        "Judge Score"
    ]
)

synthetic_usecase_data = loader.load()

# Set page_content to Description field
for doc in synthetic_usecase_data:
    doc.page_content = doc.metadata["Description"]
```

**Source**: `adv-retrieval-with-langchain.py:69-86`

**Data Structure**: Each document contains project information with metadata about domain, scores, and judge comments.

### Chain Construction Pattern

All retrieval chains in this project follow the same LCEL pattern:

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

retrieval_chain = (
    # Step 1: Retrieve context and pass through question
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    # Step 2: Assign context (allows access in next step)
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # Step 3: Generate response and preserve context
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# Invoke the chain
result = retrieval_chain.invoke({"question": "Your question here"})
response_text = result["response"].content
retrieved_docs = result["context"]
```

**Pattern Explanation**:
1. Input: Dictionary with `"question"` key
2. Retrieval: Get relevant documents using the retriever
3. Augmentation: Format prompt with question and context
4. Generation: LLM generates response
5. Output: Dictionary with `"response"` (AIMessage) and `"context"` (documents)

## Retrieval Strategies

### Overview

This project implements 7 distinct retrieval strategies, each with different strengths and use cases. All strategies can be used interchangeably in the chain pattern shown above.

### Strategy 1: Naive Retrieval (Baseline)

#### Description
Simple cosine similarity search over embedded documents. This is the baseline strategy against which all others are compared.

#### Implementation

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = Qdrant.from_documents(
    synthetic_usecase_data,
    embeddings,
    location=":memory:",
    collection_name="Synthetic_Usecases"
)

# Create retriever
naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

**Source**: `adv-retrieval-with-langchain.py:98-106`

#### Parameters
- `k` (int): Number of documents to retrieve (default: 10)
- `location` (str): Qdrant storage location (":memory:" for in-memory)
- `collection_name` (str): Name of the collection

#### Use Cases
- Baseline for comparison
- Simple semantic search requirements
- When document similarity is straightforward
- Low-latency requirements

#### Chain Example

```python
naive_retrieval_chain = (
    {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = naive_retrieval_chain.invoke({
    "question": "What is the most common project domain?"
})
```

**Source**: `adv-retrieval-with-langchain.py:194-198`

#### Advantages
- Simple to implement
- Fast execution
- Low API costs
- Good baseline performance

#### Limitations
- May miss documents with different wording but similar meaning
- No handling of query ambiguity
- Single retrieval pass

### Strategy 2: BM25 Retrieval (Lexical Search)

#### Description
Sparse retrieval using the Best Matching 25 (BM25) algorithm, based on term frequency and inverse document frequency. Excellent for exact keyword matching.

#### Implementation

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(synthetic_usecase_data)
```

**Source**: `adv-retrieval-with-langchain.py:166`

#### Parameters
- Documents are preprocessed into a bag-of-words representation
- Automatically calculates IDF scores from the document corpus
- Default `k` parameter can be adjusted

#### Use Cases
- Queries with specific terminology or jargon
- Exact phrase matching requirements
- When embedding models might miss important keywords
- Complementary to semantic search

#### Chain Example

```python
bm25_retrieval_chain = (
    {"context": itemgetter("question") | bm25_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = bm25_retrieval_chain.invoke({
    "question": "Were there any usecases about security?"
})
```

**Source**: `adv-retrieval-with-langchain.py:204-208`

#### Advantages
- Excellent for keyword/term matching
- Works well with domain-specific terminology
- No embedding required (lower cost)
- Proven algorithm with decades of research

#### Limitations
- Doesn't understand semantic similarity
- Misses paraphrased content
- Struggles with synonyms
- Requires exact or similar terms

#### When BM25 Outperforms Embeddings
- Technical documentation with specific terms
- Legal or medical texts with precise terminology
- Queries with rare or unique keywords
- When users know exact phrases to search for

### Strategy 3: Contextual Compression (Reranking)

#### Description
Two-stage retrieval: first retrieve many candidates, then rerank using a specialized model to find the most relevant. Uses Cohere's rerank-v3.5 model.

#### Implementation

```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=naive_retriever
)
```

**Source**: `adv-retrieval-with-langchain.py:169-172`

#### Parameters
- `base_compressor`: The reranking model (CohereRerank)
- `base_retriever`: Initial retriever (typically retrieves more documents)
- `model`: Cohere rerank model version

#### How It Works
1. Base retriever fetches top-k documents (e.g., 10)
2. Reranker evaluates each document against the query
3. Returns documents in new ranked order
4. Optionally filters to top-n after reranking

#### Use Cases
- When precision is critical
- Complex queries requiring nuanced understanding
- When you can afford additional API calls
- Improving recall→precision trade-off

#### Chain Example

```python
contextual_compression_retrieval_chain = (
    {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = contextual_compression_retrieval_chain.invoke({
    "question": "What did judges have to say about the fintech projects?"
})
```

**Source**: `adv-retrieval-with-langchain.py:214-218`

#### Advantages
- Significantly improves relevance
- Handles nuanced queries well
- Can rescue relevant documents from lower ranks
- State-of-the-art reranking models

#### Limitations
- Additional API cost (Cohere)
- Increased latency
- Requires two retrieval steps
- May not help if initial retrieval misses documents

#### Cost Considerations
- Cohere Rerank API charges per document reranked
- Typically rerank 10-20 documents
- Cost-effective for high-value queries

### Strategy 4: Multi-Query Retrieval

#### Description
Generates multiple reformulations of the user's query using an LLM, retrieves documents for each, and combines unique results.

#### Implementation

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever,
    llm=chat_model
)
```

**Source**: `adv-retrieval-with-langchain.py:175-177`

#### Parameters
- `retriever`: Base retriever to use for each query
- `llm`: Language model to generate query variations
- Default generates 3-5 query variations

#### How It Works
1. User submits original query
2. LLM generates multiple reformulations
3. Each reformulation queries the retriever
4. Results are combined and deduplicated
5. Unique documents returned

#### Use Cases
- Ambiguous or multi-faceted questions
- When query phrasing affects results
- Improving recall
- Complex information needs

#### Chain Example

```python
multi_query_retrieval_chain = (
    {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = multi_query_retrieval_chain.invoke({
    "question": "What is the most common project domain?"
})
```

**Source**: `adv-retrieval-with-langchain.py:224-228`

#### Example Query Expansion

Original: "What did judges say about fintech projects?"

Generated variations:
- "What were the judges' comments on financial technology projects?"
- "How did evaluators assess the fintech initiatives?"
- "What feedback was provided for finance-related projects?"

#### Advantages
- Improves recall significantly
- Handles query ambiguity
- Covers multiple perspectives
- Reduces dependency on exact phrasing

#### Limitations
- Higher API costs (LLM calls)
- Increased latency
- May retrieve less relevant documents
- More tokens for downstream processing

#### How It Improves Recall

Multi-query retrieval improves recall by:
1. Capturing different semantic angles
2. Using synonyms and paraphrases
3. Decomposing complex queries
4. Handling terminology variations

### Strategy 5: Parent Document Retrieval

#### Description
"Search small, return big" strategy. Embeds small chunks for precise matching but returns larger parent documents for complete context.

#### Implementation

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

# Set up parent documents and child splitter
parent_docs = synthetic_usecase_data
child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

# Create vector store for child chunks
client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

parent_document_vectorstore = QdrantVectorStore(
    collection_name="full_documents",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    client=client
)

# Create in-memory store for parent documents
store = InMemoryStore()

# Create retriever
parent_document_retriever = ParentDocumentRetriever(
    vectorstore=parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# Add documents (creates parent-child relationships)
parent_document_retriever.add_documents(parent_docs, ids=None)
```

**Source**: `adv-retrieval-with-langchain.py:135-160`

#### Parameters
- `vectorstore`: Store for embedded child chunks
- `docstore`: Store for parent documents (InMemoryStore)
- `child_splitter`: Text splitter for creating child chunks
- `chunk_size`: Size of child chunks (750 characters)

#### How It Works
1. Each full document is stored as a "parent" in the docstore
2. Documents are split into smaller "child chunks"
3. Child chunks are embedded and stored in vector store
4. Child chunks maintain reference to parent document
5. Search matches against child chunks
6. Returns associated parent documents

#### Use Cases
- Documents with distinct sections
- When small chunks are too narrow for context
- Precise matching with complete context needed
- Long-form content retrieval

#### Chain Example

```python
parent_document_retrieval_chain = (
    {"context": itemgetter("question") | parent_document_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = parent_document_retrieval_chain.invoke({
    "question": "Were there any usecases about security?"
})
```

**Source**: `adv-retrieval-with-langchain.py:234-238`

#### Advantages
- Precise matching via small chunks
- Complete context via parent documents
- Reduces fragmentation of information
- Better for questions needing broader context

#### Limitations
- More complex setup
- Requires two storage systems
- May return redundant information
- Higher memory usage

#### Architecture Diagram

```
Query → Embed
         ↓
    Child Chunks (Vector Store) → Find matches
         ↓
    Parent IDs
         ↓
    Parent Documents (In-Memory Store) → Return full documents
```

### Strategy 6: Ensemble Retrieval

#### Description
Combines multiple retrievers using Reciprocal Rank Fusion (RRF) algorithm. Leverages strengths of different retrieval methods.

#### Implementation

```python
from langchain.retrievers import EnsembleRetriever

retriever_list = [
    bm25_retriever,
    naive_retriever,
    parent_document_retriever,
    compression_retriever,
    multi_query_retriever
]
equal_weighting = [1/len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list,
    weights=equal_weighting
)
```

**Source**: `adv-retrieval-with-langchain.py:180-185`

#### Parameters
- `retrievers`: List of retriever instances to combine
- `weights`: List of weights for each retriever (must sum to 1.0)
- Algorithm: Reciprocal Rank Fusion (RRF)

#### How It Works
1. Each retriever returns ranked documents
2. RRF algorithm combines rankings using formula:
   ```
   RRF(d) = Σ (weight_i / (k + rank_i(d)))
   ```
   where k is a constant (typically 60)
3. Documents are re-ranked by combined score
4. Top documents returned

#### Use Cases
- When no single retriever is clearly best
- Combining lexical (BM25) and semantic (embeddings) search
- Production systems requiring robustness
- Maximizing recall across query types

#### Chain Example

```python
ensemble_retrieval_chain = (
    {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = ensemble_retrieval_chain.invoke({
    "question": "What did judges have to say about the fintech projects?"
})
```

**Source**: `adv-retrieval-with-langchain.py:244-248`

#### Custom Weighting Example

```python
# Favor BM25 for keyword-heavy queries
custom_weights = [0.4, 0.2, 0.1, 0.2, 0.1]  # Sums to 1.0

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list,
    weights=custom_weights
)
```

#### Advantages
- Combines strengths of multiple methods
- More robust across query types
- Can balance precision and recall
- Reduces individual retriever weaknesses

#### Limitations
- Highest latency (runs multiple retrievers)
- Highest cost (multiple API calls)
- Complex to tune weights
- May return duplicate documents

#### Reciprocal Rank Fusion Explained

RRF gives higher scores to documents that:
- Appear in multiple retriever results
- Rank highly in multiple retrievers
- Consistently perform well

This approach is more robust than simple score averaging because rank is more reliable than raw scores across different retrieval systems.

### Strategy 7: Semantic Chunking

#### Description
Splits documents based on semantic similarity rather than fixed character counts. Creates coherent, meaningful chunks.

#### Implementation

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# Split documents
semantic_documents = semantic_chunker.split_documents(synthetic_usecase_data[:20])

# Create vector store with semantic chunks
semantic_vectorstore = Qdrant.from_documents(
    semantic_documents,
    embeddings,
    location=":memory:",
    collection_name="Synthetic_Usecase_Data_Semantic_Chunks"
)

# Create retriever
semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
```

**Source**: `adv-retrieval-with-langchain.py:112-129`

#### Parameters
- `embeddings`: Embedding model for calculating similarity
- `breakpoint_threshold_type`: Method for determining chunk boundaries
  - `"percentile"`: Break at similarity percentile threshold
  - `"standard_deviation"`: Break based on standard deviation
  - `"interquartile"`: Break based on IQR
  - `"gradient"`: Break at similarity gradient changes

#### How It Works
1. Split document into sentences
2. Embed each sentence
3. Calculate similarity between consecutive sentences
4. Identify semantic breaks based on threshold method
5. Group sentences into coherent chunks
6. Each chunk represents a semantic unit

#### Use Cases
- Documents with clear topic transitions
- Narrative or explanatory content
- When preserving semantic coherence is critical
- Avoiding mid-topic chunk boundaries

#### Chain Example

```python
semantic_retrieval_chain = (
    {"context": itemgetter("question") | semantic_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

result = semantic_retrieval_chain.invoke({
    "question": "Were there any usecases about security?"
})
```

**Source**: `adv-retrieval-with-langchain.py:254-258`

#### Threshold Methods Comparison

**Percentile**:
```python
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # Break at 95th percentile
)
```
- Breaks at largest similarity drops
- Good for most use cases

**Standard Deviation**:
```python
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    number_of_standard_deviations=2
)
```
- Statistical approach
- Good for consistent documents

**Interquartile**:
```python
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="interquartile"
)
```
- Robust to outliers
- Good for varied content

#### Advantages
- Preserves semantic coherence
- No arbitrary mid-topic splits
- Better context preservation
- Adapts to content structure

#### Limitations
- Requires embedding all sentences (costly)
- Slower than fixed-size chunking
- Variable chunk sizes (can be very large or small)
- May not work well with lists or tables

#### When to Use Semantic Chunking

Use semantic chunking when:
- Documents have clear topic boundaries
- Content is narrative or explanatory
- Fixed-size chunks break important context
- Quality is more important than speed

Avoid when:
- Documents are short and simple
- Content is highly structured (tables, lists)
- Processing speed is critical
- Embedding costs are a concern

#### Handling Short, Repetitive Content

For FAQs or short repetitive content:

```python
# Adjust threshold to prevent over-splitting
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90  # Lower threshold
)

# Or combine semantic chunking with minimum chunk size
from langchain_text_splitters import RecursiveCharacterTextSplitter

# First semantic chunk, then ensure minimum size
semantic_chunks = semantic_chunker.split_documents(documents)

# Post-process to merge very small chunks
min_chunk_size = 200
final_chunks = []
buffer = ""

for chunk in semantic_chunks:
    if len(buffer) + len(chunk.page_content) < min_chunk_size:
        buffer += " " + chunk.page_content
    else:
        if buffer:
            final_chunks.append(buffer)
        buffer = chunk.page_content

if buffer:
    final_chunks.append(buffer)
```

## Usage Patterns

### Basic Usage

The simplest way to use any retrieval strategy:

```python
# 1. Set up environment
import os
import getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# 2. Import and initialize
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate

chat_model = ChatOpenAI(model="gpt-4.1-nano")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Load your data
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="./data/Projects_with_Domains.csv")
documents = loader.load()

# 4. Create vector store and retriever
vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 5. Build RAG chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

rag_prompt = ChatPromptTemplate.from_template(
    "Use the context to answer the question.\n\nQuestion: {question}\n\nContext: {context}"
)

chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# 6. Query
result = chain.invoke({"question": "Your question here"})
print(result["response"].content)
```

### Advanced Usage

#### Combining Multiple Strategies

```python
# Create multiple retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 1. BM25 for keywords
bm25_retriever = BM25Retriever.from_documents(documents)

# 2. Semantic search
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3. Multi-query for recall
multi_query = MultiQueryRetriever.from_llm(
    retriever=semantic_retriever,
    llm=chat_model
)

# 4. Ensemble combining BM25 and semantic
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6]
)

# 5. Add reranking on top
reranker = CohereRerank(model="rerank-v3.5")
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble
)

# Use final_retriever in chain for best results
```

#### Custom Chain with Metadata Filtering

```python
# Filter by metadata before retrieval
from langchain_core.runnables import RunnableLambda

def filter_by_domain(query_dict):
    """Filter documents by domain if specified."""
    question = query_dict["question"]
    domain = query_dict.get("domain", None)

    if domain:
        # Custom retriever with metadata filter
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {"Project Domain": domain}
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    docs = retriever.invoke(question)
    return {"context": docs, "question": question}

# Chain with filtering
filtered_chain = (
    RunnableLambda(filter_by_domain)
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# Query with domain filter
result = filtered_chain.invoke({
    "question": "What projects got high scores?",
    "domain": "Finance / FinTech"
})
```

#### Streaming Responses

```python
# Stream responses for better UX
async def stream_rag_response(question: str):
    """Stream RAG response token by token."""
    async for chunk in chain.astream({"question": question}):
        if "response" in chunk:
            # Stream the LLM response
            async for token in chunk["response"]:
                yield token.content

# Usage in async context
async for token in stream_rag_response("What are the top projects?"):
    print(token, end="", flush=True)
```

#### Batch Processing

```python
# Process multiple queries efficiently
questions = [
    "What is the most common project domain?",
    "Were there any usecases about security?",
    "What did judges say about fintech projects?"
]

# Batch invoke
results = chain.batch([{"question": q} for q in questions])

for q, result in zip(questions, results):
    print(f"Q: {q}")
    print(f"A: {result['response'].content}\n")
```

## Best Practices

### Configuration

**Environment Variables**
- Store API keys in `.env` file, never commit to version control
- Use `python-dotenv` for local development:
  ```python
  from dotenv import load_dotenv
  load_dotenv()

  openai_key = os.getenv("OPENAI_API_KEY")
  ```

**Path Management**
- Use `config.py` for all path references
- Use `pathlib.Path` for cross-platform compatibility
- Verify paths exist before processing:
  ```python
  from config import DATA_DIR
  if not DATA_DIR.exists():
      raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
  ```

**Vector Store Settings**
- Use `:memory:` for development/testing
- Use persistent storage for production:
  ```python
  # Development
  location = ":memory:"

  # Production
  location = "./qdrant_data"  # Local persistence
  # or
  location = "http://localhost:6333"  # Qdrant server
  ```

### Retrieval Strategy Selection

**Decision Tree**:

1. **Need exact keyword matching?** → Use BM25
2. **High-value queries, need maximum precision?** → Use Contextual Compression (Reranking)
3. **Queries are ambiguous or multi-faceted?** → Use Multi-Query
4. **Documents have distinct sections, need full context?** → Use Parent Document
5. **Want robust performance across query types?** → Use Ensemble
6. **Documents have clear semantic structure?** → Use Semantic Chunking
7. **Simple, fast, good enough?** → Use Naive Retrieval

**Strategy Combinations**:

| Query Type | Recommended Strategy | Alternative |
|------------|---------------------|-------------|
| Keyword-heavy | BM25 | Ensemble (BM25 + Naive) |
| Semantic search | Naive | Multi-Query |
| High precision needed | Contextual Compression | Ensemble + Reranking |
| Broad exploration | Multi-Query | Ensemble |
| Long documents | Parent Document | Semantic Chunking |
| Production (general) | Ensemble | Contextual Compression |

**Cost-Performance Trade-offs**:

| Strategy | API Calls | Latency | Cost | Precision | Recall |
|----------|-----------|---------|------|-----------|--------|
| Naive | Low | Low | $ | Medium | Medium |
| BM25 | None | Low | Free | Medium | Medium |
| Contextual Compression | High | Medium | $$$ | High | Medium |
| Multi-Query | High | Medium | $$ | Medium | High |
| Parent Document | Low | Low | $ | Medium | High |
| Ensemble | Very High | High | $$$$ | High | High |
| Semantic Chunking | High (setup) | Low (query) | $$ | Medium | Medium |

### Performance Optimization

**Caching Embeddings**
```python
# Cache embeddings to avoid recomputation
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    store,
    namespace=embeddings.model
)

# Use cached_embeddings instead of embeddings
vectorstore = Qdrant.from_documents(
    documents,
    cached_embeddings,  # Will cache and reuse
    location=":memory:",
    collection_name="cached_collection"
)
```

**Batch Embedding**
```python
# Embed documents in batches for better throughput
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=1000  # Batch size for API calls
)
```

**Limiting Retrieved Documents**
```python
# Start with fewer documents, increase if needed
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5  # Start with 5 instead of 10
    }
)

# For reranking, retrieve more initially
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20}  # Retrieve 20 for reranking
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)
```

**Async for High Throughput**
```python
# Use async methods for concurrent processing
async def process_multiple_queries(questions: list[str]):
    tasks = [chain.ainvoke({"question": q}) for q in questions]
    results = await asyncio.gather(*tasks)
    return results

# Run async
results = asyncio.run(process_multiple_queries([
    "Question 1",
    "Question 2",
    "Question 3"
]))
```

**Index Optimization**
```python
# For production, use HNSW index for faster search
from qdrant_client import models

client.create_collection(
    collection_name="optimized_collection",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        on_disk=True  # Store vectors on disk for large collections
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,  # Number of edges per node
        ef_construct=100,  # Exploration factor during construction
    )
)
```

### Monitoring and Debugging

**Enable LangSmith Tracing**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-rag-project"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"

# All chains will now be traced
result = chain.invoke({"question": "Test question"})
# View trace at: https://smith.langchain.com
```

**Log Retrieved Documents**
```python
# Log what documents are being retrieved
result = chain.invoke({"question": "Your question"})

print(f"Retrieved {len(result['context'])} documents:")
for i, doc in enumerate(result['context'], 1):
    print(f"\n{i}. {doc.metadata.get('Project Title', 'Unknown')}")
    print(f"   Score: {doc.metadata.get('Score', 'N/A')}")
    print(f"   Content: {doc.page_content[:100]}...")
```

**Measure Latency**
```python
import time

start = time.time()
result = chain.invoke({"question": "Your question"})
latency = time.time() - start

print(f"Total latency: {latency:.2f}s")
print(f"Response: {result['response'].content}")
```

## Common Patterns

### Pattern 1: Hybrid Search (BM25 + Semantic)

Combines lexical and semantic search for best of both worlds.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Create both retrievers
bm25 = BM25Retriever.from_documents(documents)
semantic = vectorstore.as_retriever(search_kwargs={"k": 10})

# Combine with 50/50 weighting
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.5, 0.5]
)

# Use in chain
hybrid_chain = (
    {"context": itemgetter("question") | hybrid_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

**When to use**: General-purpose retrieval, production systems, when query types vary

### Pattern 2: Two-Stage Retrieval (Recall + Precision)

First stage maximizes recall, second stage maximizes precision.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Stage 1: Multi-query for high recall (retrieves many documents)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
recall_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=chat_model
)

# Stage 2: Rerank for high precision
reranker = CohereRerank(model="rerank-v3.5")
precision_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=recall_retriever
)

# Use in chain
two_stage_chain = (
    {"context": itemgetter("question") | precision_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

**When to use**: Critical queries, research applications, when both recall and precision matter

### Pattern 3: Fallback Chain

Try fast retriever first, fall back to more expensive if needed.

```python
from langchain_core.runnables import RunnableFallback

# Fast retriever
fast_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Slow but thorough retriever
thorough_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
    llm=chat_model
)

# Fallback logic
def check_results(result):
    """Check if fast retrieval returned enough relevant results."""
    if len(result['context']) < 3:
        raise ValueError("Insufficient results, trying fallback")
    return result

# Chain with fallback
fallback_chain = (
    {"context": itemgetter("question") | fast_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | RunnableLambda(check_results)
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
).with_fallbacks([
    (
        {"context": itemgetter("question") | thorough_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
])
```

**When to use**: Cost optimization, when most queries are simple but some need more power

### Pattern 4: Metadata Routing

Route queries to different retrievers based on metadata or query type.

```python
from langchain_core.runnables import RunnableBranch

def route_query(query_dict):
    """Determine which retriever to use based on query."""
    question = query_dict["question"].lower()

    if any(word in question for word in ["specific", "exact", "keyword"]):
        return "bm25"
    elif any(word in question for word in ["similar", "like", "related"]):
        return "semantic"
    elif "comprehensive" in question or "all" in question:
        return "multi_query"
    else:
        return "ensemble"

# Create branch based on routing
routing_chain = RunnableBranch(
    (lambda x: route_query(x) == "bm25",
     {"context": itemgetter("question") | bm25_retriever, "question": itemgetter("question")}),
    (lambda x: route_query(x) == "semantic",
     {"context": itemgetter("question") | semantic_retriever, "question": itemgetter("question")}),
    (lambda x: route_query(x) == "multi_query",
     {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}),
    # Default to ensemble
    {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
) | RunnablePassthrough.assign(context=itemgetter("context")) \
  | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
```

**When to use**: Query-specific optimization, reducing costs while maintaining quality

## Troubleshooting

### Common Issues

#### Issue 1: Empty or Irrelevant Results

**Symptoms**: Retriever returns no documents or clearly irrelevant ones

**Causes**:
- Query and documents use different terminology
- Embedding model mismatch
- Collection is empty or incorrect

**Solutions**:
```python
# 1. Verify vector store has documents
print(f"Document count: {vectorstore.similarity_search('test', k=1)}")

# 2. Try BM25 for exact keywords
bm25_retriever = BM25Retriever.from_documents(documents)
results = bm25_retriever.invoke("your query")

# 3. Use multi-query to reformulate
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=chat_model
)

# 4. Lower similarity threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}  # Lower = more permissive
)
```

#### Issue 2: High Latency

**Symptoms**: Queries take several seconds to return

**Causes**:
- Multiple API calls (multi-query, reranking)
- Large number of documents to retrieve
- Slow embedding generation

**Solutions**:
```python
# 1. Reduce k value
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Instead of 10

# 2. Use caching
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store
)

# 3. Use async for parallel processing
results = await asyncio.gather(*[
    chain.ainvoke({"question": q}) for q in questions
])

# 4. Profile to find bottleneck
import time

start = time.time()
docs = retriever.invoke(query)
print(f"Retrieval: {time.time() - start:.2f}s")

start = time.time()
response = chat_model.invoke(prompt)
print(f"Generation: {time.time() - start:.2f}s")
```

#### Issue 3: API Rate Limits

**Symptoms**: `RateLimitError` or 429 errors

**Causes**:
- Too many concurrent requests
- Embedding large document sets
- Multi-query generating many calls

**Solutions**:
```python
# 1. Add retry logic
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def embed_with_retry(texts):
    return embeddings.embed_documents(texts)

# 2. Batch processing with delays
import time

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
    time.sleep(1)  # Rate limit protection

# 3. Use lower rate limit settings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=100,  # Smaller batches
    max_retries=3
)

# 4. Upgrade API tier or use different provider
```

#### Issue 4: Out of Memory Errors

**Symptoms**: Program crashes with memory errors, especially with large document sets

**Causes**:
- Loading entire document set into memory
- BM25 retriever storing all documents
- Parent document retriever with in-memory store

**Solutions**:
```python
# 1. Process documents in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)

# 2. Use persistent storage instead of in-memory
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_storage")  # Persistent

# 3. For BM25, limit document set
bm25_retriever = BM25Retriever.from_documents(
    documents[:1000]  # Limit to first 1000 docs
)

# 4. Use disk-based vector store
client.create_collection(
    collection_name="large_collection",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        on_disk=True  # Store on disk
    )
)
```

#### Issue 5: Inconsistent Results

**Symptoms**: Same query returns different results on repeated calls

**Causes**:
- Multi-query generating different reformulations
- LLM temperature > 0
- Ensemble retriever with variable rankings

**Solutions**:
```python
# 1. Set LLM temperature to 0 for consistency
chat_model = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0  # Deterministic
)

# 2. Fix random seed for BM25
import random
random.seed(42)

# 3. Use deterministic retrieval
naive_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

# 4. Log retrieval results for debugging
def log_retrieval(result):
    print(f"Retrieved {len(result['context'])} docs")
    for doc in result['context']:
        print(f"  - {doc.metadata.get('Project Title')}")
    return result

chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | RunnableLambda(log_retrieval)
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

#### Issue 6: High API Costs

**Symptoms**: Unexpected cloud API bills

**Causes**:
- Ensemble with many retrievers
- Multi-query generating many LLM calls
- Reranking large document sets
- Unnecessary embedding regeneration

**Solutions**:
```python
# 1. Cache embeddings aggressively
from langchain.embeddings import CacheBackedEmbeddings

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    LocalFileStore("./embedding_cache")
)

# 2. Limit ensemble components
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],  # Just 2
    weights=[0.5, 0.5]
)

# 3. Reduce reranking documents
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Rerank fewer docs
    )
)

# 4. Use cheaper models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Cheapest
chat_model = ChatOpenAI(model="gpt-4.1-nano")  # Cheaper than gpt-4

# 5. Monitor costs with LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Check cost breakdowns in LangSmith dashboard
```

## References

### Source Files

- **Configuration Module**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/config.py`
- **Main Implementation**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/adv-retrieval-with-langchain.py`
- **Interactive Notebook**: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session09-adv-retrieval.ipynb`

### External Documentation

#### LangChain
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

#### Retrieval Algorithms
- [BM25 Paper](https://www.nowpublishers.com/article/Details/INR-019)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Semantic Chunking](https://python.langchain.com/docs/how_to/semantic-chunker/)

#### Vector Databases
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)

#### APIs
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI Models](https://platform.openai.com/docs/models)
- [Cohere Rerank](https://docs.cohere.com/docs/reranking)

#### Observability
- [LangSmith](https://docs.smith.langchain.com/)
- [LangSmith Tracing](https://docs.smith.langchain.com/tracing)

### Related Notebooks

This project includes educational Jupyter notebooks:

1. **Session 7**: Synthetic Data Generation with Ragas and LangSmith
   - Path: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session07-sdg-ragas-langsmith.ipynb`
   - Topics: Creating evaluation datasets

2. **Session 8**: RAG Evaluation with Ragas
   - Path: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session08-ragas-rag-evals.ipynb`
   - Topics: Evaluating RAG system performance

3. **Session 9**: Advanced Retrieval (Main Tutorial)
   - Path: `/home/donbr/don-aie-cohort8/aie8-s09-adv-retrieval/notebooks/session09-adv-retrieval.ipynb`
   - Topics: All 7 retrieval strategies with examples

### Research Papers

- **Dense Retrieval**: [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- **Reranking**: [RankGPT: Reranking with Large Language Models](https://arxiv.org/abs/2304.09542)
- **Hybrid Search**: [Complementarity of Lexical and Semantic Matching](https://arxiv.org/abs/2104.08663)

### Community Resources

- [LangChain Discord](https://discord.gg/langchain)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Qdrant Discord](https://discord.gg/qdrant)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-08
**Maintained By**: AIE8 Advanced Retrieval Project
**License**: Project-specific (check repository root)
