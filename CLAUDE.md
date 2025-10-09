# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Session 9 repository for AI Engineering Bootcamp Cohort 8, focused on advanced retrieval methods for RAG systems using LangChain. Explores 7 retrieval strategies beyond naive similarity search, with evaluation using Ragas metrics.

**Python Version:** 3.13 (`.python-version`)

## Environment Setup

```bash
# Install dependencies and sync environment
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/WSL/Mac

# Launch Jupyter
jupyter notebook
```

**Required Environment Variables:**
```bash
export OPENAI_API_KEY="your-key-here"          # Required
export COHERE_API_KEY="your-key-here"          # Required for reranking
export LANGCHAIN_API_KEY="your-key-here"       # Optional (LangSmith tracing)
export LANGCHAIN_TRACING_V2="true"             # Optional (enable tracing)
export LANGCHAIN_PROJECT="your-project-name"   # Optional (LangSmith project)
```

## Key Technologies

- **LangChain + Experimental**: RAG chains, retrievers, semantic chunking
- **OpenAI**: `gpt-4.1-nano` (LLM), `text-embedding-3-small` (embeddings)
- **Cohere**: `rerank-v3.5` model for contextual compression
- **Qdrant**: In-memory vector database
- **Ragas**: RAG evaluation framework (v0.2.10)
- **rank-bm25**: Sparse retrieval implementation

## Core Retrieval Implementations

All 7 retrieval strategies share the same LCEL chain pattern with only the retriever component changing:

```python
retrieval_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

### Retrieval Strategies

1. **Naive Retrieval** - `vectorstore.as_retriever(search_kwargs={"k": 10})`
2. **BM25** - `BM25Retriever.from_documents(documents)` (sparse, keyword-based)
3. **Contextual Compression (Reranking)** - `ContextualCompressionRetriever(base_compressor=CohereRerank(), base_retriever=...)`
4. **Multi-Query** - `MultiQueryRetriever.from_llm(retriever=..., llm=...)` (generates query variations)
5. **Parent Document** - `ParentDocumentRetriever(vectorstore=..., docstore=InMemoryStore(), child_splitter=...)` (search small, return big)
6. **Ensemble** - `EnsembleRetriever(retrievers=[...], weights=[...])` (Reciprocal Rank Fusion)
7. **Semantic Chunking** - `SemanticChunker(embeddings, breakpoint_threshold_type="percentile")` (preprocessing, not a retriever)

### Critical Parameters

- **k=10**: Used across retrievers to provide sufficient documents for reranking
- **Embedding dimension**: 1536 for `text-embedding-3-small` (must match in manual Qdrant setup)
- **LLM**: `gpt-4.1-nano` chosen to focus evaluation on retrieval quality, not generation

## Data Architecture

**Primary Dataset:** `data/Projects_with_Domains.csv`
- AI project use cases with domains, descriptions, judge comments
- CSV columns become document metadata
- `page_content` manually set to `Description` field for embedding

**Document Preparation Pattern:**
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="./data/Projects_with_Domains.csv",
    metadata_columns=["Project Title", "Project Domain", "Description", ...]
)
documents = loader.load()

# CRITICAL: Set page_content to the field you want embedded
for doc in documents:
    doc.page_content = doc.metadata["Description"]
```

## Qdrant Vector Store Patterns

**Pattern 1: High-level API (most retrievers)**
```python
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="collection_name"
)
```

**Pattern 2: Manual client setup (Parent Document Retriever)**
```python
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="collection_name",
    vectors_config=models.VectorParams(
        size=1536,  # MUST match embedding dimension
        distance=models.Distance.COSINE
    )
)
vectorstore = QdrantVectorStore(
    collection_name="collection_name",
    embedding=embeddings,
    client=client
)
```

## Assignment Structure

**Official Assignment:** `Advanced_Retrieval_with_LangChain_Assignment.ipynb`

**Breakout Room Part 1:**
- Task 1-3: Setup (dependencies, data, Qdrant)
- Task 4-10: Implement 7 retrieval strategies
- Answer embedded questions in markdown cells

**Breakout Room Part 2 (Activity #1):**
1. Create "golden dataset" using Ragas Synthetic Data Generation
2. Evaluate each retriever with Ragas retriever-specific metrics
3. Write comparison paragraph analyzing:
   - **Cost** (track via LangSmith)
   - **Latency** (track via LangSmith)
   - **Performance** (Ragas metrics: context precision, recall, relevance)

**Submission Workflow:**
```bash
# Create assignment branch (NOT main)
git checkout -b s09-assignment

# Complete work and commit
git add Advanced_Retrieval_with_LangChain_Assignment.ipynb
git commit -m "Complete Session 9 advanced retrieval assignment"

# Push to assignment branch
git push -u origin s09-assignment
```

**Deliverables:**
- GitHub URL to notebook on assignment branch
- 5-minute Loom video walkthrough
- 3 lessons learned / 3 lessons not yet learned
- Social media posts (optional, extra credit)

## Repository Analyzer Framework

This repository includes a portable, drop-in analysis toolkit (`ra_*` directories) for multi-domain repository analysis.

### Framework Structure

```
ra_orchestrators/          # Domain orchestrators
├── base_orchestrator.py   # Core framework (343 lines)
├── architecture_orchestrator.py
└── ux_orchestrator.py

ra_agents/                 # Agent definitions (JSON)
├── architecture/
│   ├── analyzer.json
│   └── doc_writer.json
└── ux/
    ├── ux_researcher.json
    └── ui_designer.json

ra_tools/                  # Tool integrations
├── mcp_registry.py        # MCP server discovery
└── figma_integration.py   # Figma MCP + REST API

ra_output/                 # Timestamped analysis outputs
└── {domain}_{YYYYMMDD_HHMMSS}/
```

### Running Orchestrators

```bash
# Architecture analysis
python -m ra_orchestrators.architecture_orchestrator

# UX design workflow
python -m ra_orchestrators.ux_orchestrator "Project Name"

# With timeout for long-running analyses
timeout 1800 python -m ra_orchestrators.architecture_orchestrator
```

**Output:** Each run creates `ra_output/{domain}_{timestamp}/` with timestamped subdirectories to prevent overwrites.

### BaseOrchestrator Pattern

All orchestrators inherit from `BaseOrchestrator` (ra_orchestrators/base_orchestrator.py:30):

```python
from ra_orchestrators.base_orchestrator import BaseOrchestrator

class CustomOrchestrator(BaseOrchestrator):
    def __init__(self):
        super().__init__(
            domain_name="custom",
            output_base_dir=Path("ra_output"),
            use_timestamp=True
        )

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        return {"agent_name": AgentDefinition(...)}

    def get_allowed_tools(self) -> List[str]:
        return ["Read", "Write", "Grep", "Glob", "Bash"]

    async def run(self):
        await self.execute_phase("phase_1", "agent_name", "Task...", self.client)
```

### Agent Registry

Agents are defined as JSON files in `ra_agents/{domain}/{agent_name}.json`:

```python
from ra_agents.registry import AgentRegistry

registry = AgentRegistry()

# Discover all agents
all_agents = registry.discover_agents()

# Discover domain-specific agents
ux_agents = registry.discover_agents(domain="ux")

# Load specific agent
agent = registry.load_agent("analyzer", domain="architecture")

# Load all agents for a domain
agents = registry.load_domain_agents("ux")
```

**Agent JSON Structure:**
```json
{
  "name": "agent_name",
  "description": "What this agent does",
  "prompt": "Detailed system prompt...",
  "tools": ["Read", "Write", "Grep", "Glob"],
  "model": "sonnet",
  "domain": "custom",
  "version": "1.0.0"
}
```

## Project Configuration

**config.py** provides centralized paths and settings:

```python
from config import PROJECT_ROOT, DATA_DIR, QDRANT_COLLECTION_NAME

# Access configured paths
csv_path = DATA_DIR / "Projects_with_Domains.csv"
```

**Available configurations:**
- `PROJECT_ROOT`: Repository root path
- `DATA_DIR`: Data directory (`data/`)
- `NOTEBOOKS_DIR`: Notebooks directory (`notebooks/`)
- `DEFAULT_CHUNK_SIZE`: 500
- `DEFAULT_CHUNK_OVERLAP`: 50
- `QDRANT_COLLECTION_NAME`: "Use Case RAG"
- `QDRANT_LOCATION`: ":memory:"

## Exploration Notebooks

**notebooks/** contains learning progression (Sessions 07-09):

1. **session07-sdg-ragas-langsmith.ipynb** - Synthetic Data Generation using Ragas, LangSmith integration
2. **session08-ragas-rag-evals.ipynb** - RAG evaluation framework, generates `baseline_evaluation_dataset.csv` and `rerank_evaluation_dataset.csv`
3. **session09-adv-retrieval.ipynb** - Consolidated implementation of all 7 retrieval methods with sample queries

**Workflow:** Work through session notebooks sequentially to understand the full evaluation pipeline before tackling the assignment.

## Module Code Organization

**adv-retrieval.py** consolidates retrieval implementations for reuse across notebooks:

```python
# Import in notebooks
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

# Use retriever implementations from module
```

## Retrieval Strategy Trade-offs

**When Each Excels:**
- **BM25**: Exact keyword matching, technical queries
- **Contextual Compression**: High-precision needs, cost-sensitive (fewer tokens to LLM)
- **Multi-Query**: Broad coverage, ambiguous queries
- **Parent Document**: Need surrounding context, avoid chopped semantics
- **Ensemble**: Maximum robustness across query types
- **Semantic Chunking**: Clean semantic breaks in corpus (not useful for short/repetitive text)

**Trade-offs:**
- **BM25**: Fast, no embeddings needed, but weak on semantic similarity
- **Reranking**: Adds latency + Cohere API cost
- **Multi-Query**: Multiple LLM calls = slower + more expensive
- **Parent Document**: Risk of too much irrelevant context
- **Ensemble**: Complex to tune and debug

## Advanced Build (Optional)

Alternative to standard assignment: Implement [RAG-Fusion](https://arxiv.org/pdf/2402.03367) using LangChain ecosystem. Same submission requirements apply.

## Additional Resources

- LangChain retrievers: https://python.langchain.com/docs/how_to/#retrievers
- Semantic chunking: "The 5 Levels of Text Splitting for Retrieval" (Greg Kamradt)
- ChunkViz: https://chunkviz.up.railway.app/
- Ragas documentation: https://docs.ragas.io/
