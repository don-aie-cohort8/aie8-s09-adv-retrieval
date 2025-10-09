# %% [markdown]
# # Synthetic Data Generation Using RAGAS - RAG Evaluation with LangSmith

# %%
# Standard Library Imports
import getpass
import os
from operator import itemgetter
from uuid import uuid4

# Third-Party Imports
# LangChain Core
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# LangChain Community
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant

# LangChain OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import apply_transforms, default_transforms

# LangSmith
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate

# Local Application Imports
# (none yet)


# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
os.environ["LANGCHAIN_PROJECT"] = f"AIM - SDG - {uuid4().hex[0:8]}"
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# %%
BASELINE_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

DOPE_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Make your answer rad, ensure high levels of dopeness. Do not be generic, or give generic responses.

Context: {context}
Question: {question}
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

docs_copy = docs

# %%
# RAGAS Models
# LangchainEmbeddingsWrapper are deprecated
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
transformer_llm = generator_llm
embedding_model = generator_embeddings

# LangChain Models
llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
eval_llm = ChatOpenAI(model="gpt-4.1")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# %% [markdown]
# ## RAGAS - knowledge graph approach

# %%
kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

# %%
kg_transform = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)

# %%
apply_transforms(kg, kg_transform)

# %%
kg.save("usecase_data_kg.json")

# %%
usecase_data_kg = KnowledgeGraph.load("usecase_data_kg.json")

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=usecase_data_kg)

query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]

# %%
golden_testset = generator.generate(testset_size=10, query_distribution=query_distribution)
golden_testset.to_pandas()

# %% [markdown]
# ## LangChain RAG

# %% [markdown]
# ### First Chain - baseline_chain

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

# redefine rag_documents
baseline_documents = text_splitter.split_documents(docs_copy)

# %%
baseline_vectorstore = Qdrant.from_documents(
    documents=baseline_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Use Case RAG"
)

# %%
baseline_retriever = baseline_vectorstore.as_retriever(search_kwargs={"k": 10})

baseline_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

baseline_chain = (
    {"context": itemgetter("question") | baseline_retriever, "question": itemgetter("question")}
    | baseline_prompt | llm | StrOutputParser()
)

# %%
baseline_chain.invoke({"question" : "What are people doing with AI these days?"})

# %% [markdown]
# ### Second Chain - dope_chain

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50
)

# redefine rag_documents
dope_documents = text_splitter.split_documents(docs_copy)

# %%
# reuse of vectorstore name for different collection
# will this reuse the existing in-memory instance or overwrite?
dope_vectorstore = Qdrant.from_documents(
    documents=dope_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Use Case RAG Docs"
)

# %%
dope_prompt = ChatPromptTemplate.from_template(DOPE_PROMPT)
dope_retriever = dope_vectorstore.as_retriever()

dope_chain = (
    {"context": itemgetter("question") | dope_retriever, "question": itemgetter("question")}
    | dope_prompt | llm | StrOutputParser()
)

# %%
dope_chain.invoke({"question" : "How are people using AI to make money?"})

# %% [markdown]
# ## LangSmith

# %% [markdown]
# ### Create Dataset - from RAGAS Golden Testset

# %%
client = Client()

langsmith_dataset_name = f"Use Case Synthetic Data - AIE8 - {uuid4().hex[0:8]}"

langsmith_dataset = client.create_dataset(
    dataset_name=langsmith_dataset_name,
    description="Synthetic Data for Use Cases"
)

# %%
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

# %% [markdown]
# ### Setup Evaluation Criteria - using legacy approach... not OpenEvals

# %%
qa_evaluator = LangChainStringEvaluator("qa", config={"llm" : eval_llm})

labeled_helpfulness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "helpfulness": (
                "Is this submission helpful to the user,"
                " taking into account the correct reference answer?"
            )
        },
        "llm" : eval_llm
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    }
)

dopeness_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "dopeness": "Is this response dope, lit, cool, or is it just a generic response?",
        },
        "llm" : eval_llm
    }
)

# %% [markdown]
# ## Run Evaluations

# %% [markdown]
# ### First Evaluation - baseline_chain

# %%
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

# %% [markdown]
# ### Second Evaluation - dope_chain

# %%
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

# %% [markdown]
# ## Python Library Versions - from uv.lock

# %% [markdown]
# 
# ```
# [package.metadata]
# requires-dist = [
#     { name = "jupyter", specifier = ">=1.1.1" },
#     { name = "langchain-community", specifier = ">=0.3.16" },
#     { name = "langchain-openai", specifier = ">=0.3.3" },
#     { name = "langchain-qdrant", specifier = ">=0.2.0" },
#     { name = "langgraph", specifier = ">=0.2.69" },
#     { name = "nltk", specifier = "==3.8.1" },
#     { name = "numpy", specifier = ">=2.2.2" },
#     { name = "pymupdf", specifier = ">=1.26.3" },
#     { name = "ragas", specifier = "==0.2.10" },
#     { name = "unstructured", specifier = ">=0.14.8" },
# ]
# ```

# %% [markdown]
# 


