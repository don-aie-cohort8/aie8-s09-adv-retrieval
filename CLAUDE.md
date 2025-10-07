# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Session 9 repository for AI Engineering Bootcamp Cohort 8, focused on advanced retrieval methods for RAG (Retrieval-Augmented Generation) systems using LangChain. This session explores techniques to improve retrieval performance beyond naive similarity search.

## Python Environment Setup

```bash
# Install dependencies using uv
uv sync

# Launch Jupyter notebook
jupyter notebook
```

**Python Version:** 3.13 (specified in `.python-version`)

## Key Technologies

### Core Stack
- **LangChain**: Orchestration framework for RAG chains and retrievers
- **LangChain Experimental**: Semantic chunking implementation
- **OpenAI**: LLM (gpt-4.1-nano) and embeddings (text-embedding-3-small)
- **Cohere**: Reranking API (rerank-v3.5 model)
- **Qdrant**: Vector database (in-memory mode for assignments)
- **Ragas**: RAG evaluation framework

### Retrieval Methods Implemented
1. **Naive Retrieval** — Basic cosine similarity with k=10
2. **BM25** — Sparse retrieval using bag-of-words (rank-bm25)
3. **Contextual Compression (Reranking)** — Cohere rerank-v3.5 for result compression
4. **Multi-Query Retrieval** — LLM-generated query variations for broader coverage
5. **Parent Document Retrieval** — Small-to-big strategy (search small chunks, return large context)
6. **Ensemble Retrieval** — Reciprocal Rank Fusion across multiple retrievers
7. **Semantic Chunking** — Embedding-based sentence clustering (percentile threshold)

## Data

- **Source**: `data/Projects_with_Domains.csv` — AI project use cases with domains, descriptions, judge comments
- **Alternative**: `data/howpeopleuseai.pdf` — Available but not used in primary notebook

## Common Development Commands

### Running the Notebook
```bash
# Start Jupyter and work through the assignment notebook
jupyter notebook Advanced_Retrieval_with_LangChain_Assignment.ipynb
```

### Environment Variables Required
```bash
export OPENAI_API_KEY="your-key-here"
export COHERE_API_KEY="your-key-here"
export LANGCHAIN_API_KEY="optional-for-tracing"  # For LangSmith observability
```

## Assignment Workflow

### Standard Assignment (Main Build)
1. Complete breakout room activities in `Advanced_Retrieval_with_LangChain_Assignment.ipynb`:
   - **Part 1**: Implement 7 retrieval strategies (Tasks 1-10)
   - **Part 2**: Evaluate retrievers using Ragas metrics (Activity #1)
2. Answer embedded questions in markdown cells
3. Create evaluation comparing retrievers on cost, latency, and performance
4. Submit on assignment branch (NOT main)

### Optional Advanced Build
Implement RAG-Fusion (https://arxiv.org/pdf/2402.03367) using LangChain ecosystem as alternative to standard assignment.

### Submission Requirements
```bash
# Create assignment branch
git checkout -b s09-assignment

# Complete work and commit
git add Advanced_Retrieval_with_LangChain_Assignment.ipynb
git commit -m "Complete Session 9 advanced retrieval assignment"

# Push to assignment branch
git push -u origin s09-assignment
```

Include in submission:
- GitHub URL to notebook on assignment branch
- 5-minute Loom video walkthrough
- 3 lessons learned / 3 lessons not yet learned
- Social media posts (optional, extra credit)

## Architecture Patterns

### Retriever Chain Structure
All retrieval chains follow the same LCEL pattern with only the retriever changing:

```python
retrieval_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

### Qdrant Vector Store Setup
Two patterns used:

**Pattern 1: High-level API**
```python
vectorstore = Qdrant.from_documents(
    documents, embeddings,
    location=":memory:",
    collection_name="collection_name"
)
```

**Pattern 2: Manual client setup (for Parent Document Retriever)**
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="collection_name",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)
vectorstore = QdrantVectorStore(collection_name="collection_name", embedding=embeddings, client=client)
```

**Note**: Embedding dimension (1536) matches text-embedding-3-small. Change if using different model.

### Evaluation Strategy (Breakout Room Part 2)

Activity 1 requires:
1. **Golden Dataset Creation**: Use Ragas Synthetic Data Generation (or similar)
2. **Retriever Evaluation**: Apply Ragas retriever-specific metrics to each method
3. **Analysis**: Write paragraph comparing retrievers factoring in:
   - **Cost**: Track via LangSmith
   - **Latency**: Monitor via LangSmith
   - **Performance**: Ragas metrics (context precision, recall, relevance)

**Important**: Plan before coding. Semantic chunking is not a retriever (it's preprocessing) but can be compared as "on vs off."

## Key Design Decisions

### Document Preparation
CSV loader stores all metadata fields; `page_content` is manually set to `Description` field for embedding:
```python
for doc in synthetic_usecase_data:
    doc.page_content = doc.metadata["Description"]
```

### Retrieval Parameters
- **k=10**: Used across retrievers to provide sufficient documents for reranking
- **Model Choice**: gpt-4.1-nano (lightweight) to focus evaluation on retrieval quality, not generation
- **Embedding Model**: text-embedding-3-small for cost efficiency

### Parent Document Strategy
- **Parent docs**: Full CSV rows (no splitting at parent level)
- **Child chunks**: RecursiveCharacterTextSplitter with chunk_size=750
- **Storage**: InMemoryStore for parents, Qdrant for child embeddings

### Ensemble Configuration
Equal weighting across all 5 retrievers: `[1/5, 1/5, 1/5, 1/5, 1/5]`

## Notes on Retrieval Methods

### When Each Excels
- **BM25**: Exact keyword matching, technical queries
- **Contextual Compression**: High-precision needs, cost-sensitive applications (fewer tokens to LLM)
- **Multi-Query**: Broad coverage, ambiguous queries
- **Parent Document**: Need surrounding context, avoid chopped semantics
- **Ensemble**: Maximum robustness across query types
- **Semantic Chunking**: Clean semantic breaks in corpus (not useful for short/repetitive text like FAQs)

### Trade-offs
- **BM25**: Fast, no embeddings needed, but weak on semantic similarity
- **Reranking**: Adds latency + Cohere API cost
- **Multi-Query**: Multiple LLM calls = slower + more expensive
- **Parent Document**: Risk of too much irrelevant context
- **Ensemble**: Complex to tune and debug

## Additional Resources

Referenced in session materials:
- LangChain retrievers documentation: https://python.langchain.com/docs/how_to/#retrievers
- Semantic chunking video: "The 5 Levels of Text Splitting for Retrieval" (Greg Kamradt)
- ChunkViz visualization tool: https://chunkviz.up.railway.app/
- AI Makerspace events on Advanced Retrieval and Semantic Chunking (YouTube)
