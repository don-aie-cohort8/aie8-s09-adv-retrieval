# %% [markdown]
# # Using Ragas to Evaluate a RAG Application

# %%
# Standard Library Imports
import copy
import os
import time
from getpass import getpass
from uuid import uuid4

# Third-Party Imports
# LangChain Core
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# LangChain Community
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

# LangChain Integrations
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# LangGraph
from langgraph.graph import START, StateGraph

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# RAGAS
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextEntityRecall,
    Faithfulness,
    FactualCorrectness,
    LLMContextRecall,
    NoiseSensitivity,
    ResponseRelevancy,
)
from ragas.testset import TestsetGenerator

# Typing
from typing_extensions import List, TypedDict

# Local Application Imports
# (none yet)

# %%
os.environ["OPENAI_API_KEY"] = getpass("Please enter your OpenAI API key!")
os.environ["COHERE_API_KEY"] = getpass("Please enter your Cohere API key!")
os.environ["LANGSMITH_API_KEY"] = getpass("LangSmith API Key:")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = f"AIM - RAGAS EVALS - {uuid4().hex[0:8]}"

# %%
BASELINE_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""

# %%
from pathlib import Path

project_root = Path.cwd().parent  # Go up one level from notebooks/ to project root
data_path = project_root / "data"

print(f"Project root: {project_root}")
print(f"Data path: {data_path}")
print(f"Data path exists: {data_path.exists()}")

# Load documents
loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

ragas_docs = docs
retriever_docs = docs

# %%
# RAGAS
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

# langchain_openai
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4.1-nano")

# %%
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# %%
golden_testset = generator.generate_with_langchain_docs(ragas_docs, testset_size=10)
golden_testset.to_pandas()

# %%
baseline_dataset = copy.deepcopy(golden_testset)
rerank_dataset = copy.deepcopy(golden_testset)

# %% [markdown]
# ## LangGraph RAG

# %% [markdown]
# ### First Graph - baseline

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
split_documents = text_splitter.split_documents(docs)
len(split_documents)

# %%
rag_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

# %%
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

# %%
_ = baseline_vector_store.add_documents(documents=split_documents)

retriever = baseline_vector_store.as_retriever(search_kwargs={"k": 3})

# %%
def retrieve(state):
  retrieved_docs = retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

# %%
def generate(state):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = llm.invoke(messages)
  return {"response" : response.content}

# %%
class State(TypedDict):
  question: str
  context: List[Document]
  response: str

# %%
baseline_graph_builder = StateGraph(State).add_sequence([retrieve, generate])
baseline_graph_builder.add_edge(START, "retrieve")
baseline_graph = baseline_graph_builder.compile()

# %% [markdown]
# ### Second Graph - Reranker

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
split_documents = text_splitter.split_documents(retriever_docs)
len(split_documents)

# %%
rerank_client = QdrantClient(":memory:")

rerank_client.create_collection(
    collection_name="use_case_data_new_chunks",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

rerank_vector_store = QdrantVectorStore(
    client=rerank_client,
    collection_name="use_case_data_new_chunks",
    embedding=embeddings,
)

# %%
_ = rerank_vector_store.add_documents(documents=split_documents)

baseline_retriever = rerank_vector_store.as_retriever(search_kwargs={"k": 20})

# %%
def retrieve_reranked(state):
  compressor = CohereRerank(model="rerank-v3.5")
  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=baseline_retriever, search_kwargs={"k": 5}
  )
  retrieved_docs = compression_retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

# %%
class AdjustedState(TypedDict):
  question: str
  context: List[Document]
  response: str

rerank_graph_builder = StateGraph(AdjustedState).add_sequence([retrieve_reranked, generate])
rerank_graph_builder.add_edge(START, "retrieve_reranked")
rerank_graph = rerank_graph_builder.compile()

# %% [markdown]
# ## RAGAS

# %%
custom_run_config = RunConfig(timeout=360)

# %% [markdown]
# ### Raw Evaluation Data

# %%
for test_row in baseline_dataset:
  response = baseline_graph.invoke({"question" : test_row.eval_sample.user_input})
  test_row.eval_sample.response = response["response"]
  test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]

# %%
baseline_evaluation_dataset = EvaluationDataset.from_pandas(baseline_dataset.to_pandas())

# %%
baseline_evaluation_dataset.to_csv("baseline_evaluation_dataset.csv")

# %%
for test_row in rerank_dataset:
  response = rerank_graph.invoke({"question" : test_row.eval_sample.user_input})
  test_row.eval_sample.response = response["response"]
  test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
  time.sleep(2) # To try to avoid rate limiting.

# %%
rerank_evaluation_dataset = EvaluationDataset.from_pandas(rerank_dataset.to_pandas())

# %%
rerank_evaluation_dataset.to_csv("rerank_evaluation_dataset.csv")

# %% [markdown]
# ### Evaluation Results

# %%
baseline_result = evaluate(
    dataset=baseline_evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=custom_run_config
)

# %%
rerank_evaluation_result = evaluate(
    dataset=rerank_evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=custom_run_config
)

# %% [markdown]
# ## Compare Evaluation Results

# %%
baseline_result

# %%
rerank_evaluation_result

# %% [markdown]
# ## Python Library Versions - from uv.lock

# %% [markdown]
# ```
# requires-dist = [
#     { name = "claude-agent-sdk", specifier = ">=0.1.0" },
#     { name = "cohere", specifier = ">=5.12.0,<5.13.0" },
#     { name = "ipykernel", specifier = ">=6.30.1" },
#     { name = "jupyter", specifier = ">=1.1.1" },
#     { name = "langchain", specifier = ">=0.3.14" },
#     { name = "langchain-cohere", specifier = "==0.4.4" },
#     { name = "langchain-community", specifier = ">=0.3.29" },
#     { name = "langchain-openai", specifier = ">=0.3.33" },
#     { name = "langchain-qdrant", specifier = ">=0.2.1" },
#     { name = "langgraph", specifier = "==0.6.7" },
#     { name = "pymupdf", specifier = ">=1.24.0" },
#     { name = "pyppeteer", specifier = ">=2.0.0" },
#     { name = "qdrant-client", specifier = ">=1.7.0" },
#     { name = "ragas", specifier = "==0.2.10" },
#     { name = "rapidfuzz", specifier = ">=3.0.0" },
# ]
# ```

# %% [markdown]
# 

