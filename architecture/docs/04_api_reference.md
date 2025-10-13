# API Reference

## Overview

This API reference documents the AIE8 Advanced Retrieval project, which demonstrates various RAG (Retrieval-Augmented Generation) patterns including synthetic data generation, evaluation frameworks, and seven advanced retrieval strategies using LangChain, RAGAS, and LangSmith.

The project is organized into three main sessions:
- **Session 07**: Synthetic data generation using RAGAS with LangSmith integration
- **Session 08**: RAG evaluation with baseline and reranked retrieval using RAGAS metrics
- **Session 09**: Seven advanced retrieval strategies with performance comparisons

## Configuration Module

### config.py

Central configuration file for project paths and settings.

**Source**: `config.py`

#### Configuration Variables

- **PROJECT_ROOT: Path**
  - **Source**: `config.py:9`
  - **Description**: Root directory of the project
  - **Value**: Dynamically resolved from `Path(__file__).parent`
  - **Example**:
    ```python
    from config import PROJECT_ROOT
    print(PROJECT_ROOT)  # /home/user/aie8-s09-adv-retrieval
    ```

- **DATA_DIR: Path**
  - **Source**: `config.py:10`
  - **Description**: Directory containing project data files (PDFs, CSVs)
  - **Value**: `PROJECT_ROOT / "data"`
  - **Example**:
    ```python
    from config import DATA_DIR
    print(DATA_DIR.exists())  # True
    ```

- **NOTEBOOKS_DIR: Path**
  - **Source**: `config.py:11`
  - **Description**: Directory containing Jupyter notebooks
  - **Value**: `PROJECT_ROOT / "notebooks"`

- **DOCS_DIR: Path**
  - **Source**: `config.py:12`
  - **Description**: Directory for documentation files
  - **Value**: `PROJECT_ROOT / "docs"`

- **PDF_FILES: Path**
  - **Source**: `config.py:15`
  - **Description**: Pattern for locating PDF files
  - **Value**: `DATA_DIR / "*.pdf"`

- **CSV_FILES: Path**
  - **Source**: `config.py:16`
  - **Description**: Pattern for locating CSV files
  - **Value**: `DATA_DIR / "*.csv"`

#### Model Configuration

- **DEFAULT_CHUNK_SIZE: int**
  - **Source**: `config.py:19`
  - **Description**: Default chunk size for text splitting
  - **Value**: `500`
  - **Example**:
    ```python
    from config import DEFAULT_CHUNK_SIZE
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=50
    )
    ```

- **DEFAULT_CHUNK_OVERLAP: int**
  - **Source**: `config.py:20`
  - **Description**: Default overlap between chunks
  - **Value**: `50`

#### Vector Store Configuration

- **QDRANT_COLLECTION_NAME: str**
  - **Source**: `config.py:23`
  - **Description**: Default Qdrant collection name
  - **Value**: `"Use Case RAG"`
  - **Example**:
    ```python
    from config import QDRANT_COLLECTION_NAME
    from langchain_community.vectorstores import Qdrant

    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name=QDRANT_COLLECTION_NAME
    )
    ```

- **QDRANT_LOCATION: str**
  - **Source**: `config.py:24`
  - **Description**: Qdrant storage location (in-memory by default)
  - **Value**: `":memory:"`

#### Environment Variables

The project requires the following environment variables:

- **OPENAI_API_KEY** - OpenAI API key for LLM and embeddings
- **COHERE_API_KEY** - Cohere API key for reranking (Sessions 08 & 09)
- **LANGCHAIN_API_KEY** / **LANGSMITH_API_KEY** - LangSmith API key for tracing and evaluation
- **LANGCHAIN_TRACING_V2** - Enable LangSmith tracing (set to `"true"`)
- **LANGCHAIN_PROJECT** / **LANGSMITH_PROJECT** - Project name for LangSmith tracking

**Example Setup**:
```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")
os.environ["COHERE_API_KEY"] = getpass("Cohere API Key:")
os.environ["LANGSMITH_API_KEY"] = getpass("LangSmith API Key:")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "My-RAG-Project"
```

## Session 07: Synthetic Data Generation

### Overview

Session 07 demonstrates synthetic data generation using RAGAS (Retrieval-Augmented Generation Assessment) to create test datasets for RAG evaluation. It includes knowledge graph construction, test set generation with multiple query types, and LangSmith dataset creation.

**Source**: `session07-sdg-ragas-langsmith.py`

### Key Components

#### RAGAS Knowledge Graph

The session uses RAGAS's knowledge graph approach to generate high-quality synthetic test data.

**Knowledge Graph Construction**:
```python
# Source: session07-sdg-ragas-langsmith.py:107-114
from ragas.testset.graph import KnowledgeGraph, Node, NodeType

kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
```

**Apply Transformations**:
```python
# Source: session07-sdg-ragas-langsmith.py:117-120
from ragas.testset.transforms import apply_transforms, default_transforms

kg_transform = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, kg_transform)
```

**Save and Load Knowledge Graph**:
```python
# Source: session07-sdg-ragas-langsmith.py:123-126
kg.save("usecase_data_kg.json")
usecase_data_kg = KnowledgeGraph.load("usecase_data_kg.json")
```

#### RAGAS Test Set Generator

**Model Configuration**:
```python
# Source: session07-sdg-ragas-langsmith.py:92-101
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS Models
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
transformer_llm = generator_llm
embedding_model = generator_embeddings

# LangChain Models
llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
eval_llm = ChatOpenAI(model="gpt-4.1")
```

**Test Set Generation**:
```python
# Source: session07-sdg-ragas-langsmith.py:128-137
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=usecase_data_kg)

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]

golden_testset = generator.generate(testset_size=10, query_distribution=query_distribution)
df = golden_testset.to_pandas()
```

#### RAG Chains

**Baseline Chain**:
```python
# Source: session07-sdg-ragas-langsmith.py:147-171
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

baseline_documents = text_splitter.split_documents(docs_copy)

baseline_vectorstore = Qdrant.from_documents(
    documents=baseline_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Use Case RAG"
)

baseline_retriever = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})

BASELINE_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

baseline_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

baseline_chain = (
    {"context": itemgetter("question") | baseline_retriever, "question": itemgetter("question")}
    | baseline_prompt | llm | StrOutputParser()
)
```

**Dope Chain (Alternative Configuration)**:
```python
# Source: session07-sdg-ragas-langsmith.py:180-205
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

dope_documents = text_splitter.split_documents(docs_copy)

dope_vectorstore = Qdrant.from_documents(
    documents=dope_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Use Case RAG Docs"
)

DOPE_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Make your answer rad, ensure high levels of dopeness. Do not be generic, or give generic responses.

Context: {context}
Question: {question}
"""

dope_prompt = ChatPromptTemplate.from_template(DOPE_PROMPT)
dope_retriever = dope_vectorstore.as_retriever()

dope_chain = (
    {"context": itemgetter("question") | dope_retriever, "question": itemgetter("question")}
    | dope_prompt | llm | StrOutputParser()
)
```

### LangSmith Integration

#### Creating Datasets

```python
# Source: session07-sdg-ragas-langsmith.py:217-239
from langsmith import Client
from uuid import uuid4

client = Client()

langsmith_dataset_name = f"Use Case Synthetic Data - AIE8 - {uuid4().hex[0:8]}"

langsmith_dataset = client.create_dataset(
    dataset_name=langsmith_dataset_name,
    description="Synthetic Data for Use Cases"
)

# Add examples from RAGAS testset
for data_row in golden_testset.to_pandas().iterrows():
    client.create_example(
        inputs={
            "question": data_row[1]["user_input"]
        },
        outputs={
            "answer": data_row[1]["reference"]
        },
        metadata={
            "context": data_row[1]["reference_contexts"]
        },
        dataset_id=langsmith_dataset.id
    )
```

#### LangSmith Evaluators

```python
# Source: session07-sdg-ragas-langsmith.py:245-273
from langsmith.evaluation import LangChainStringEvaluator

# QA Evaluator
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm})

# Helpfulness Evaluator
labeled_helpfulness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "helpfulness": (
                "Is this submission helpful to the user,"
                " taking into account the correct reference answer?"
            )
        },
        "llm": eval_llm
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    }
)

# Dopeness Evaluator
dopeness_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "dopeness": "Is this response dope, lit, cool, or is it just a generic response?",
        },
        "llm": eval_llm
    }
)
```

#### Running Evaluations

```python
# Source: session07-sdg-ragas-langsmith.py:282-306
from langsmith.evaluation import evaluate

# Evaluate baseline chain
evaluate(
    baseline_chain.invoke,
    data=langsmith_dataset_name,
    evaluators=[
        qa_evaluator,
        labeled_helpfulness_evaluator,
        dopeness_evaluator
    ],
    metadata={"revision_id": "default_chain_init"},
)

# Evaluate dope chain
evaluate(
    dope_chain.invoke,
    data=langsmith_dataset_name,
    evaluators=[
        qa_evaluator,
        labeled_helpfulness_evaluator,
        dopeness_evaluator
    ],
    metadata={"revision_id": "dope_chain"},
)
```

### Usage Examples

#### Complete SDG Workflow

```python
# 1. Load documents
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from pathlib import Path

project_root = Path.cwd().parent
data_path = project_root / "data"

loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

# 2. Create knowledge graph
from ragas.testset.graph import KnowledgeGraph, Node, NodeType

kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

# 3. Apply transformations
from ragas.testset.transforms import apply_transforms, default_transforms

kg_transform = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, kg_transform)

# 4. Generate testset
generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=kg)
golden_testset = generator.generate(testset_size=10, query_distribution=query_distribution)

# 5. Create LangSmith dataset
client = Client()
dataset = client.create_dataset(dataset_name="My-Dataset", description="Test data")

for data_row in golden_testset.to_pandas().iterrows():
    client.create_example(
        inputs={"question": data_row[1]["user_input"]},
        outputs={"answer": data_row[1]["reference"]},
        metadata={"context": data_row[1]["reference_contexts"]},
        dataset_id=dataset.id
    )
```

### Best Practices

- **Query Distribution**: Use a mix of query types (50% single-hop, 25% multi-hop abstract, 25% multi-hop specific) for comprehensive coverage
- **Knowledge Graph Persistence**: Save knowledge graphs to JSON files for reusability and faster iteration
- **Model Selection**: Use cost-effective models for generation (gpt-4.1-nano) and more powerful models for evaluation (gpt-4.1)
- **Chunk Size Tuning**: Experiment with different chunk sizes (500-1000 characters) based on document structure
- **LangSmith Projects**: Use unique project names with UUIDs for easy tracking and comparison

## Session 08: RAG Evaluation

### Overview

Session 08 demonstrates RAG evaluation using RAGAS metrics with two approaches: baseline retrieval and reranked retrieval. It includes LangGraph workflow construction, RAGAS evaluation with 6 metrics, and comprehensive performance comparison.

**Source**: `session08-ragas-rag-evals.py`

### State Classes

#### State (TypedDict)

- **Source**: `session08-ragas-rag-evals.py:157-160`
- **Description**: TypedDict defining the state structure for the baseline RAG graph
- **Fields**:
  - `question: str` - User's input question
  - `context: List[Document]` - Retrieved document contexts
  - `response: str` - Generated answer from the LLM
- **Example**:
  ```python
  from typing_extensions import TypedDict, List
  from langchain_core.documents import Document

  class State(TypedDict):
      question: str
      context: List[Document]
      response: str

  # Usage in graph
  initial_state = {"question": "What are AI use cases?"}
  ```

#### AdjustedState (TypedDict)

- **Source**: `session08-ragas-rag-evals.py:204-207`
- **Description**: TypedDict defining the state structure for the reranked RAG graph (identical to State)
- **Fields**:
  - `question: str` - User's input question
  - `context: List[Document]` - Reranked document contexts
  - `response: str` - Generated answer from the LLM
- **Example**:
  ```python
  class AdjustedState(TypedDict):
      question: str
      context: List[Document]
      response: str
  ```

### Functions

#### retrieve(state: State) -> State

- **Source**: `session08-ragas-rag-evals.py:145-147`
- **Description**: Retrieves relevant documents for a given question using vector similarity search
- **Parameters**:
  - `state: State` - Dictionary containing the user question
- **Returns**: `State` - Updated state with retrieved context documents
- **Implementation Details**:
  - Uses Qdrant vector store with OpenAI embeddings (text-embedding-3-small)
  - Retrieves top 3 documents (k=3)
  - Collection: "use_case_data"
  - Distance metric: COSINE
- **Example**:
  ```python
  from langchain_qdrant import QdrantVectorStore
  from qdrant_client import QdrantClient
  from qdrant_client.http.models import Distance, VectorParams
  from langchain_openai import OpenAIEmbeddings

  # Setup vector store
  baseline_client = QdrantClient(":memory:")
  baseline_client.create_collection(
      collection_name="use_case_data",
      vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
  )

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  baseline_vector_store = QdrantVectorStore(
      client=baseline_client,
      collection_name="use_case_data",
      embedding=embeddings,
  )

  baseline_vector_store.add_documents(documents=split_documents)
  retriever = baseline_vector_store.as_retriever(search_kwargs={"k": 3})

  # Retrieve function
  def retrieve(state):
      retrieved_docs = retriever.invoke(state["question"])
      return {"context": retrieved_docs}

  # Usage
  state = {"question": "What are AI applications?"}
  result = retrieve(state)
  print(len(result["context"]))  # 3
  ```

#### generate(state: State) -> State

- **Source**: `session08-ragas-rag-evals.py:150-154`
- **Description**: Generates an answer using the retrieved context and the user's question
- **Parameters**:
  - `state: State` - Dictionary containing question and retrieved context
- **Returns**: `State` - Updated state with generated response
- **Implementation Details**:
  - Uses ChatOpenAI with gpt-4.1-nano model
  - Concatenates document content with double newlines
  - Formats prompt with question and context
  - Returns only the response content (not the full message)
- **Example**:
  ```python
  from langchain_openai import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate

  BASELINE_PROMPT = """\
  You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

  ### Question
  {question}

  ### Context
  {context}
  """

  llm = ChatOpenAI(model="gpt-4.1-nano")
  rag_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

  def generate(state):
      docs_content = "\n\n".join(doc.page_content for doc in state["context"])
      messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
      response = llm.invoke(messages)
      return {"response": response.content}

  # Usage
  state = {
      "question": "What is AI?",
      "context": [Document(page_content="AI is artificial intelligence...")]
  }
  result = generate(state)
  print(result["response"])
  ```

#### retrieve_reranked(state: AdjustedState) -> AdjustedState

- **Source**: `session08-ragas-rag-evals.py:195-201`
- **Description**: Retrieves and reranks documents using Cohere's reranking model for improved relevance
- **Parameters**:
  - `state: AdjustedState` - Dictionary containing the user question
- **Returns**: `AdjustedState` - Updated state with reranked context documents
- **Implementation Details**:
  - Initial retrieval: top 20 documents from vector store
  - Reranker: Cohere rerank-v3.5 model
  - Final output: top 5 reranked documents
  - Uses ContextualCompressionRetriever wrapper
- **Example**:
  ```python
  from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
  from langchain_cohere import CohereRerank

  # Setup base retriever with higher k
  baseline_retriever = rerank_vector_store.as_retriever(search_kwargs={"k": 20})

  # Create reranking retriever
  def retrieve_reranked(state):
      compressor = CohereRerank(model="rerank-v3.5")
      compression_retriever = ContextualCompressionRetriever(
          base_compressor=compressor,
          base_retriever=baseline_retriever,
          search_kwargs={"k": 5}
      )
      retrieved_docs = compression_retriever.invoke(state["question"])
      return {"context": retrieved_docs}

  # Usage
  state = {"question": "What are AI use cases?"}
  result = retrieve_reranked(state)
  print(len(result["context"]))  # 5 reranked documents
  ```

### LangGraph Workflows

#### Baseline RAG Graph

```python
# Source: session08-ragas-rag-evals.py:163-165
from langgraph.graph import START, StateGraph

baseline_graph_builder = StateGraph(State).add_sequence([retrieve, generate])
baseline_graph_builder.add_edge(START, "retrieve")
baseline_graph = baseline_graph_builder.compile()

# Usage
response = baseline_graph.invoke({"question": "What is AI?"})
print(response["response"])
```

**Flow**: START → retrieve → generate

#### Reranked RAG Graph

```python
# Source: session08-ragas-rag-evals.py:209-211
rerank_graph_builder = StateGraph(AdjustedState).add_sequence([retrieve_reranked, generate])
rerank_graph_builder.add_edge(START, "retrieve_reranked")
rerank_graph = rerank_graph_builder.compile()

# Usage
response = rerank_graph.invoke({"question": "What is AI?"})
print(response["response"])
```

**Flow**: START → retrieve_reranked → generate

### RAGAS Evaluation

#### Metrics

The session evaluates RAG systems using 6 RAGAS metrics:

1. **LLMContextRecall** - Measures if all ground truth information is present in retrieved context
2. **Faithfulness** - Measures if the answer is faithful to the retrieved context
3. **FactualCorrectness** - Evaluates factual accuracy of the generated answer
4. **ResponseRelevancy** - Measures how relevant the response is to the question
5. **ContextEntityRecall** - Measures entity-level recall in retrieved context
6. **NoiseSensitivity** - Evaluates robustness to noisy/irrelevant context

```python
# Source: session08-ragas-rag-evals.py:38-45
from ragas.metrics import (
    ContextEntityRecall,
    Faithfulness,
    FactualCorrectness,
    LLMContextRecall,
    NoiseSensitivity,
    ResponseRelevancy,
)
```

#### Running Evaluations

```python
# Source: session08-ragas-rag-evals.py:217-256
from ragas import EvaluationDataset, RunConfig, evaluate
import time

# Configure run settings
custom_run_config = RunConfig(timeout=360)

# Prepare baseline evaluation dataset
for test_row in baseline_dataset:
    response = baseline_graph.invoke({"question": test_row.eval_sample.user_input})
    test_row.eval_sample.response = response["response"]
    test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]

baseline_evaluation_dataset = EvaluationDataset.from_pandas(baseline_dataset.to_pandas())
baseline_evaluation_dataset.to_csv("baseline_evaluation_dataset.csv")

# Prepare rerank evaluation dataset
for test_row in rerank_dataset:
    response = rerank_graph.invoke({"question": test_row.eval_sample.user_input})
    test_row.eval_sample.response = response["response"]
    test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
    time.sleep(2)  # Avoid rate limiting

rerank_evaluation_dataset = EvaluationDataset.from_pandas(rerank_dataset.to_pandas())
rerank_evaluation_dataset.to_csv("rerank_evaluation_dataset.csv")

# Evaluate baseline
baseline_result = evaluate(
    dataset=baseline_evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ],
    llm=evaluator_llm,
    run_config=custom_run_config
)

# Evaluate reranked
rerank_evaluation_result = evaluate(
    dataset=rerank_evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ],
    llm=evaluator_llm,
    run_config=custom_run_config
)
```

### Usage Examples

#### Complete Evaluation Pipeline

```python
# 1. Setup environment
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")
os.environ["COHERE_API_KEY"] = getpass("Cohere API Key:")
os.environ["LANGSMITH_API_KEY"] = getpass("LangSmith API Key:")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "RAG-Evaluation"

# 2. Load and split documents
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

project_root = Path.cwd().parent
data_path = project_root / "data"

loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
split_documents = text_splitter.split_documents(docs)

# 3. Create vector store
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

baseline_client = QdrantClient(":memory:")
baseline_client.create_collection(
    collection_name="use_case_data",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

baseline_vector_store = QdrantVectorStore(
    client=baseline_client,
    collection_name="use_case_data",
    embedding=embeddings,
)

baseline_vector_store.add_documents(documents=split_documents)

# 4. Generate test data
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
golden_testset = generator.generate_with_langchain_docs(docs, testset_size=10)

# 5. Build baseline graph
from langgraph.graph import START, StateGraph

baseline_graph_builder = StateGraph(State).add_sequence([retrieve, generate])
baseline_graph_builder.add_edge(START, "retrieve")
baseline_graph = baseline_graph_builder.compile()

# 6. Prepare evaluation dataset
import copy

baseline_dataset = copy.deepcopy(golden_testset)

for test_row in baseline_dataset:
    response = baseline_graph.invoke({"question": test_row.eval_sample.user_input})
    test_row.eval_sample.response = response["response"]
    test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]

# 7. Evaluate
from ragas import EvaluationDataset, RunConfig, evaluate

baseline_evaluation_dataset = EvaluationDataset.from_pandas(baseline_dataset.to_pandas())

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
custom_run_config = RunConfig(timeout=360)

baseline_result = evaluate(
    dataset=baseline_evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ],
    llm=evaluator_llm,
    run_config=custom_run_config
)

print(baseline_result)
```

### Best Practices

- **Document Chunking**: Use chunk_size=500, chunk_overlap=30 for balanced context and performance
- **Retrieval Configuration**: Start with k=3 for baseline, k=20 for reranking input
- **Reranking**: Use Cohere rerank-v3.5 and reduce to top 5 documents for final context
- **Evaluation Timeout**: Set RunConfig timeout to 360 seconds to avoid API timeouts
- **Rate Limiting**: Add sleep delays (2 seconds) between evaluation runs to avoid rate limits
- **Model Selection**: Use gpt-4.1-nano for generation (cost-effective), gpt-4.1-mini for evaluation (balanced)
- **Dataset Persistence**: Save evaluation datasets to CSV for reproducibility
- **LangSmith Tracing**: Enable tracing to track costs, latency, and debug issues

## Session 09: Advanced Retrieval Strategies

### Overview

Session 09 demonstrates seven advanced retrieval strategies using LangChain, comparing their performance, cost, and latency characteristics. Each strategy is implemented with a consistent LCEL chain pattern for easy comparison.

**Source**: `session09-adv-retrieval.py`

### Retrieval Strategies

#### 1. Naive Retrieval (Cosine Similarity)

- **Source**: `session09-adv-retrieval.py:98-106`
- **Description**: Standard vector similarity search using cosine distance
- **Configuration**:
  - Embedding model: text-embedding-3-small (OpenAI)
  - Top K: 10
  - Vector store: Qdrant (in-memory)
  - Distance metric: Cosine similarity
- **When to Use**:
  - General purpose retrieval
  - When documents have clear semantic patterns
  - Baseline for comparison
- **Example**:
  ```python
  from langchain_community.vectorstores import Qdrant
  from langchain_openai import OpenAIEmbeddings

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  vectorstore = Qdrant.from_documents(
      synthetic_usecase_data,
      embeddings,
      location=":memory:",
      collection_name="Synthetic_Usecases"
  )

  naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

  # Use in chain
  from operator import itemgetter
  from langchain_core.runnables import RunnablePassthrough
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI

  chat_model = ChatOpenAI(model="gpt-4.1-nano")

  RAG_TEMPLATE = """\
  You are a helpful and kind assistant. Use the context provided below to answer the question.

  If you do not know the answer, or are unsure, say you don't know.

  Query:
  {question}

  Context:
  {context}
  """

  rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

  naive_retrieval_chain = (
      {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = naive_retrieval_chain.invoke({"question": "What is the most common project domain?"})
  print(result["response"].content)
  ```

#### 2. BM25 (Sparse Retrieval)

- **Source**: `session09-adv-retrieval.py:166-167`
- **Description**: Sparse retrieval using Best Matching 25 (BM25) algorithm based on bag-of-words
- **Configuration**:
  - Algorithm: BM25 (Okapi BM25)
  - No embedding required (term-based)
  - Returns documents ranked by term frequency and inverse document frequency
- **When to Use**:
  - Keyword-heavy queries
  - Exact term matching is important
  - Complement to semantic search
  - Lower computational cost than embeddings
- **Example**:
  ```python
  from langchain_community.retrievers import BM25Retriever

  bm25_retriever = BM25Retriever.from_documents(synthetic_usecase_data)

  bm25_retrieval_chain = (
      {"context": itemgetter("question") | bm25_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = bm25_retrieval_chain.invoke({"question": "Were there any usecases about security?"})
  print(result["response"].content)
  ```

#### 3. Contextual Compression (Reranking)

- **Source**: `session09-adv-retrieval.py:169-172`
- **Description**: Two-stage retrieval that first retrieves many documents, then reranks them for relevance
- **Configuration**:
  - Base retriever: Naive retriever (k=10)
  - Reranker: Cohere rerank-v3.5
  - Final output: Compressed/reranked subset
- **When to Use**:
  - Quality over quantity
  - When willing to trade latency for precision
  - Complex queries requiring nuanced relevance
  - After initial retrieval needs refinement
- **Example**:
  ```python
  from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
  from langchain_cohere import CohereRerank

  compressor = CohereRerank(model="rerank-v3.5")
  compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor,
      base_retriever=naive_retriever
  )

  contextual_compression_retrieval_chain = (
      {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = contextual_compression_retrieval_chain.invoke({"question": "What did judges have to say about the fintech projects?"})
  print(result["response"].content)
  ```

#### 4. Multi-Query Retrieval

- **Source**: `session09-adv-retrieval.py:175-177`
- **Description**: Generates multiple query variations using an LLM, retrieves for each, and combines unique results
- **Configuration**:
  - Base retriever: Naive retriever
  - Query generator: ChatOpenAI (gpt-4.1-nano)
  - Generates ~3-5 query variations automatically
  - Returns union of all retrieved documents
- **When to Use**:
  - Complex or ambiguous queries
  - Improving recall
  - When queries can be interpreted multiple ways
  - Overcoming vocabulary mismatch
- **Example**:
  ```python
  from langchain.retrievers.multi_query import MultiQueryRetriever

  multi_query_retriever = MultiQueryRetriever.from_llm(
      retriever=naive_retriever,
      llm=chat_model
  )

  multi_query_retrieval_chain = (
      {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = multi_query_retrieval_chain.invoke({"question": "What is the most common project domain?"})
  print(result["response"].content)
  ```

#### 5. Parent Document Retrieval

- **Source**: `session09-adv-retrieval.py:135-160`
- **Description**: "Small-to-big" strategy that searches small chunks but returns larger parent documents
- **Configuration**:
  - Child splitter: RecursiveCharacterTextSplitter(chunk_size=750)
  - Parent documents: Original unsplit documents
  - Vector store: Qdrant for child chunks
  - Doc store: InMemoryStore for parent documents
  - Embedding: text-embedding-3-small (1536 dimensions)
- **When to Use**:
  - Need precise retrieval with broad context
  - Documents have logical parent-child structure
  - Want to avoid cutting off relevant surrounding information
  - Balancing precision and context window
- **Example**:
  ```python
  from langchain.retrievers import ParentDocumentRetriever
  from langchain.storage import InMemoryStore
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from qdrant_client import QdrantClient, models
  from langchain_qdrant import QdrantVectorStore

  parent_docs = synthetic_usecase_data
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

  # Setup vector store
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

  # Create retriever
  store = InMemoryStore()

  parent_document_retriever = ParentDocumentRetriever(
      vectorstore=parent_document_vectorstore,
      docstore=store,
      child_splitter=child_splitter,
  )

  # Add documents
  parent_document_retriever.add_documents(parent_docs, ids=None)

  # Use in chain
  parent_document_retrieval_chain = (
      {"context": itemgetter("question") | parent_document_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = parent_document_retrieval_chain.invoke({"question": "Were there any usecases about security?"})
  print(result["response"].content)
  ```

#### 6. Ensemble Retrieval

- **Source**: `session09-adv-retrieval.py:180-185`
- **Description**: Combines multiple retrievers using Reciprocal Rank Fusion (RRF) algorithm
- **Configuration**:
  - Retrievers combined (5 total):
    1. BM25 (sparse)
    2. Naive vector retrieval
    3. Parent document retrieval
    4. Contextual compression (reranked)
    5. Multi-query retrieval
  - Weights: Equal weighting (0.2 each)
  - Fusion method: Reciprocal Rank Fusion
- **When to Use**:
  - Want best of all worlds
  - Different retrieval strategies excel at different queries
  - Willing to accept higher latency for better quality
  - Production systems requiring robustness
- **Example**:
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

  ensemble_retrieval_chain = (
      {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = ensemble_retrieval_chain.invoke({"question": "What did judges have to say about the fintech projects?"})
  print(result["response"].content)
  ```

#### 7. Semantic Chunking

- **Source**: `session09-adv-retrieval.py:112-129`
- **Description**: Chunks documents based on semantic similarity rather than fixed character counts
- **Configuration**:
  - Embedding model: text-embedding-3-small
  - Breakpoint threshold: percentile
  - Algorithm: Embeds sentences, groups by semantic similarity
  - Sample size: First 20 documents (configurable)
- **When to Use**:
  - Documents with natural semantic breaks
  - Want chunks that preserve complete thoughts
  - Document structure is semantically meaningful
  - Prefer quality over speed in chunking
- **Example**:
  ```python
  from langchain_experimental.text_splitter import SemanticChunker
  from langchain_openai import OpenAIEmbeddings

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  semantic_chunker = SemanticChunker(
      embeddings,
      breakpoint_threshold_type="percentile"
  )

  # Split documents semantically
  semantic_documents = semantic_chunker.split_documents(synthetic_usecase_data[:20])

  # Create vector store
  semantic_vectorstore = Qdrant.from_documents(
      semantic_documents,
      embeddings,
      location=":memory:",
      collection_name="Synthetic_Usecase_Data_Semantic_Chunks"
  )

  semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})

  # Use in chain
  semantic_retrieval_chain = (
      {"context": itemgetter("question") | semantic_retriever, "question": itemgetter("question")}
      | RunnablePassthrough.assign(context=itemgetter("context"))
      | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
  )

  result = semantic_retrieval_chain.invoke({"question": "What is the most common project domain?"})
  print(result["response"].content)
  ```

### LCEL Chain Pattern

All retrieval strategies use a consistent LCEL (LangChain Expression Language) chain pattern:

```python
# Source: session09-adv-retrieval.py:194-198
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

retrieval_chain = (
    # Step 1: Extract question and retrieve context
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    # Step 2: Pass context through (preserves for output)
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # Step 3: Generate response and preserve context
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# Invoke with question
result = retrieval_chain.invoke({"question": "What is AI?"})
print(result["response"].content)  # Generated answer
print(len(result["context"]))      # Retrieved documents
```

**Pattern Benefits**:
- Consistent interface across all strategies
- Preserves context for inspection/debugging
- Easy to swap retriever implementations
- Supports LangSmith tracing

### Usage Examples

#### Comparing Retrieval Strategies

```python
# Define test questions
questions = [
    "What is the most common project domain?",
    "Were there any usecases about security?",
    "What did judges have to say about the fintech projects?"
]

# Compare all strategies
strategies = {
    "Naive": naive_retrieval_chain,
    "BM25": bm25_retrieval_chain,
    "Reranked": contextual_compression_retrieval_chain,
    "Multi-Query": multi_query_retrieval_chain,
    "Parent Document": parent_document_retrieval_chain,
    "Ensemble": ensemble_retrieval_chain,
    "Semantic": semantic_retrieval_chain,
}

results = {}
for strategy_name, chain in strategies.items():
    results[strategy_name] = []
    for question in questions:
        response = chain.invoke({"question": question})
        results[strategy_name].append({
            "question": question,
            "answer": response["response"].content,
            "num_docs": len(response["context"])
        })

# Print comparison
for strategy_name, strategy_results in results.items():
    print(f"\n=== {strategy_name} ===")
    for result in strategy_results:
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:100]}...")
        print(f"Docs: {result['num_docs']}\n")
```

#### Custom Retriever Configuration

```python
# Create custom naive retriever with different k
custom_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create custom BM25 with specific parameters
from langchain_community.retrievers import BM25Retriever

custom_bm25 = BM25Retriever.from_documents(
    synthetic_usecase_data,
    k=15  # Return more documents
)

# Create custom reranker with different model
from langchain_cohere import CohereRerank

custom_compressor = CohereRerank(
    model="rerank-v3.5",
    top_n=3  # Return fewer documents
)
custom_compression_retriever = ContextualCompressionRetriever(
    base_compressor=custom_compressor,
    base_retriever=custom_retriever
)

# Create custom ensemble with different weights
custom_ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, naive_retriever, compression_retriever],
    weights=[0.5, 0.3, 0.2]  # Favor BM25
)

# Use in chain
custom_chain = (
    {"context": itemgetter("question") | custom_ensemble, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

### Best Practices

#### Strategy Selection

**Use Naive Retrieval when**:
- Starting out / establishing baseline
- Documents are semantically well-structured
- Speed is critical
- Cost constraints are tight

**Use BM25 when**:
- Queries contain specific keywords or terminology
- Exact matching is important
- No embedding infrastructure available
- Complementing semantic search

**Use Reranking when**:
- Quality is more important than speed
- Initial retrieval returns too many marginally relevant docs
- Have budget for additional API calls
- Complex queries requiring nuanced relevance

**Use Multi-Query when**:
- Queries are ambiguous or complex
- Need to improve recall
- Vocabulary mismatch between query and documents
- Can afford extra LLM calls

**Use Parent Document when**:
- Documents have natural hierarchical structure
- Need precise search but broader context
- Chunks are too small but whole documents too large
- Want best of both worlds (precision + context)

**Use Ensemble when**:
- Production system requiring robustness
- Different query types need different strategies
- Can afford higher latency
- Want to hedge against any single strategy's weaknesses

**Use Semantic Chunking when**:
- Documents have clear semantic structure
- Quality of chunks is critical
- Initial chunking quality impacts retrieval
- Can afford slower preprocessing

#### Performance Trade-offs

**Latency** (Fastest to Slowest):
1. Naive Retrieval (~100-200ms)
2. BM25 (~100-200ms)
3. Semantic Chunking (~100-300ms)
4. Parent Document (~200-400ms)
5. Multi-Query (~500-1000ms, multiple LLM calls)
6. Reranking (~500-800ms, additional API call)
7. Ensemble (~1-2s, combines multiple strategies)

**Cost** (Cheapest to Most Expensive):
1. BM25 (no API calls)
2. Naive Retrieval (embedding calls only)
3. Semantic Chunking (embedding calls)
4. Parent Document (embedding calls)
5. Multi-Query (+ LLM calls for query generation)
6. Reranking (+ Cohere API calls)
7. Ensemble (sum of all component costs)

**Quality** (General observations):
- Reranking typically provides highest precision
- Ensemble provides best overall robustness
- Multi-Query improves recall significantly
- BM25 excels at keyword matching
- Parent Document balances precision and context

#### Configuration Tips

**Optimal Settings for Different Use Cases**:

```python
# High-precision, cost-aware
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = CohereRerank(model="rerank-v3.5", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# High-recall, comprehensive search
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
    llm=chat_model
)

# Balanced performance
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vectorstore.as_retriever(search_kwargs={"k": 10})
    ],
    weights=[0.4, 0.6]  # Favor semantic
)

# Production-ready robust system
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        naive_retriever,
        compression_retriever
    ],
    weights=[0.3, 0.3, 0.4]  # Favor reranked
)
```

## Common Patterns

### RAG Pipeline Pattern

The standard RAG pipeline follows this pattern across all sessions:

```python
# 1. Load documents
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

# 2. Split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 3. Create vector store
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Qdrant.from_documents(
    split_docs,
    embeddings,
    location=":memory:",
    collection_name="MyCollection"
)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 5. Create prompt
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer based on context: {context}
Question: {question}
""")

# 6. Create chain
from langchain_openai import ChatOpenAI
from operator import itemgetter

llm = ChatOpenAI(model="gpt-4.1-nano")

chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt | llm
)

# 7. Invoke
response = chain.invoke({"question": "What is AI?"})
```

### LangSmith Tracing

Enable comprehensive tracing and monitoring:

```python
import os
from uuid import uuid4

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # or LANGSMITH_TRACING
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"  # or LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = f"My-Project-{uuid4().hex[0:8]}"

# Tracing is now automatic for all LangChain calls
result = chain.invoke({"question": "What is AI?"})

# View in LangSmith UI:
# - Token usage
# - Latency breakdown
# - Cost per call
# - Input/output inspection
# - Error tracking
```

### Vector Store Setup

Standard pattern for Qdrant vector store:

```python
# Method 1: Simple (in-memory)
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="MyCollection"
)

# Method 2: Advanced (with client)
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="MyCollection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="MyCollection",
    embedding=embeddings
)

vectorstore.add_documents(documents)
```

## Dependencies

### Required Packages

Based on the project's dependency management:

**Core LangChain**:
- `langchain>=0.3.14` - Core LangChain framework
- `langchain-core` - Core abstractions
- `langchain-community>=0.3.16` - Community integrations
- `langgraph>=0.2.69` - Graph-based workflows

**LLM Providers**:
- `langchain-openai>=0.3.3` - OpenAI integration (LLMs & embeddings)
- `langchain-cohere==0.4.4` - Cohere integration (reranking)

**Vector Stores**:
- `langchain-qdrant>=0.2.0` - Qdrant vector database integration
- `qdrant-client>=1.7.0` - Qdrant Python client

**Evaluation & Testing**:
- `ragas==0.2.10` - RAG evaluation framework

**Document Processing**:
- `pymupdf>=1.24.0` - PDF document loading
- `unstructured>=0.14.8` - Advanced document parsing

**Utilities**:
- `jupyter>=1.1.1` - Jupyter notebook support
- `ipykernel>=6.30.1` - Jupyter kernel
- `numpy>=2.2.2` - Numerical operations
- `nltk==3.8.1` - Natural language processing

**Other**:
- `cohere>=5.12.0` - Direct Cohere API access
- `rapidfuzz>=3.0.0` - Fast fuzzy string matching

### API Keys Required

- **OPENAI_API_KEY** - OpenAI API key
  - Used for: LLMs (gpt-4.1-nano, gpt-4.1-mini, gpt-4.1) and embeddings (text-embedding-3-small, text-embedding-3-large)
  - Get from: https://platform.openai.com/api-keys

- **COHERE_API_KEY** - Cohere API key
  - Used for: Reranking with rerank-v3.5 model
  - Get from: https://dashboard.cohere.com/api-keys

- **LANGSMITH_API_KEY** / **LANGCHAIN_API_KEY** - LangSmith API key
  - Used for: Tracing, evaluation, dataset management
  - Get from: https://smith.langchain.com/settings

## Troubleshooting

### Common Issues

**Issue**: `ImportError: cannot import name 'Qdrant' from 'langchain_community.vectorstores'`
- **Solution**: Ensure you have the correct version installed: `pip install langchain-community>=0.3.16`

**Issue**: Rate limiting errors from OpenAI
- **Solution**: Add delays between calls: `time.sleep(2)` or use async with rate limiting

**Issue**: Qdrant collection already exists error
- **Solution**: Use unique collection names or clear existing: `client.delete_collection("collection_name")`

**Issue**: RAGAS evaluation timeout
- **Solution**: Increase timeout in RunConfig: `RunConfig(timeout=360)`

**Issue**: Cohere reranking API errors
- **Solution**: Verify API key is set correctly and check Cohere dashboard for usage limits

**Issue**: LangSmith traces not appearing
- **Solution**: Ensure both `LANGSMITH_TRACING="true"` and `LANGSMITH_API_KEY` are set

**Issue**: Memory errors with large document sets
- **Solution**:
  - Process documents in batches
  - Use disk-based Qdrant instead of `:memory:`
  - Reduce chunk sizes or document count

**Issue**: Inconsistent retrieval results
- **Solution**:
  - Set random seed for reproducibility
  - Ensure embeddings are consistent
  - Check for documents being modified between runs

## Performance Considerations

### Latency by Strategy

**Single Query Latency** (approximate):
- Naive Retrieval: 100-200ms
- BM25: 100-200ms
- Semantic Chunking: 100-300ms
- Parent Document: 200-400ms
- Reranking: 500-800ms
- Multi-Query: 500-1000ms
- Ensemble: 1-2 seconds

**Factors Affecting Latency**:
- Network latency to API providers
- Document corpus size
- Embedding model size
- Number of documents retrieved (k parameter)
- LLM model size and speed

### Token Consumption

**Approximate Tokens per Operation**:

**Embedding Operations**:
- Document chunking: ~500 tokens per chunk (average)
- Query embedding: ~10-50 tokens per query

**LLM Operations**:
- Simple generation (gpt-4.1-nano): 500-1000 tokens total
- Evaluation (gpt-4.1-mini): 1000-2000 tokens per eval
- Multi-query generation: +500-1000 tokens per query
- RAGAS evaluation: 2000-5000 tokens per test case

**Cost Optimization**:
- Use gpt-4.1-nano for generation ($0.30/1M input tokens)
- Use text-embedding-3-small for embeddings ($0.02/1M tokens)
- Cache embeddings when possible
- Use smaller k values for retrieval
- Batch evaluate rather than individual calls

### Memory Usage

**Vector Store Memory**:
- In-memory Qdrant: ~1-2 MB per 1000 documents (with embeddings)
- Disk-based Qdrant: Minimal memory footprint

**Document Processing**:
- Raw documents: Varies by content
- Split documents: ~2x raw document size
- Embeddings: 1536 dimensions × 4 bytes = ~6KB per document

**Optimization Tips**:
- Use disk-based storage for large corpora
- Process documents in batches
- Clear vector stores when switching collections
- Use generators for large document sets
- Monitor memory with `memory_profiler`

## References

- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [BM25 Algorithm Paper](https://www.nowpublishers.com/article/Details/INR-019)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-08
**Project**: AIE8 Advanced Retrieval (Session 07-09)
