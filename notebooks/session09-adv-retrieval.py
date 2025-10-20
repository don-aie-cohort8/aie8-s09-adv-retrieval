# %% [markdown]
# # Advanced Retrieval with LangChain

# %%
# Standard Library Imports
import getpass
import os
from datetime import datetime, timedelta
from operator import itemgetter
from uuid import uuid4

# Third-Party Imports
# LangChain Core
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# LangChain Community
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant

# LangChain Integrations
from langchain_cohere import CohereRerank
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Qdrant
from qdrant_client import QdrantClient, models

# Local Application Imports
# (none yet)


# %%
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIM - SDG - {uuid4().hex[0:8]}"

# %%
chat_model = ChatOpenAI(model="gpt-4.1-nano")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# %%
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# %%
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
        "Judge Score",
    ],
)

synthetic_usecase_data = loader.load()

for doc in synthetic_usecase_data:
    doc.page_content = doc.metadata["Description"]

# %%
synthetic_usecase_data[0]

# %% [markdown]
# ## Vector Stores

# %% [markdown]
# ### Naive Vector Store

# %%
vectorstore = Qdrant.from_documents(
    synthetic_usecase_data,
    embeddings,
    location=":memory:",
    collection_name="Synthetic_Usecases",
)

# %%
naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# %% [markdown]
# ### Semantic Vector Store

# %%
semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

# %%
semantic_documents = semantic_chunker.split_documents(synthetic_usecase_data[:20])

# %%
semantic_vectorstore = Qdrant.from_documents(
    semantic_documents,
    embeddings,
    location=":memory:",
    collection_name="Synthetic_Usecase_Data_Semantic_Chunks",
)

# %%
semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})

# %% [markdown]
# ### Parent Document Vector Store

# %%
parent_docs = synthetic_usecase_data
child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

# %%
client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)

parent_document_vectorstore = QdrantVectorStore(
    collection_name="full_documents",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    client=client,
)

# %%
store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# %%
parent_document_retriever.add_documents(parent_docs, ids=None)

# %% [markdown]
# ### Other Retrievers

# %%
bm25_retriever = BM25Retriever.from_documents(synthetic_usecase_data)

# %%
compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=naive_retriever
)

# %%
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever, llm=chat_model
)

# %%
retriever_list = [
    bm25_retriever,
    naive_retriever,
    parent_document_retriever,
    compression_retriever,
    multi_query_retriever,
]
equal_weighting = [1 / len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list, weights=equal_weighting
)

# %% [markdown]
# ## LangChain Retrieval Chains

# %% [markdown]
# ### Naive

# %%
naive_retrieval_chain = (
    {
        "context": itemgetter("question") | naive_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ### BM25

# %%
bm25_retrieval_chain = (
    {
        "context": itemgetter("question") | bm25_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ### Contextual Compression

# %%
contextual_compression_retrieval_chain = (
    {
        "context": itemgetter("question") | compression_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ### Multi-Query

# %%
multi_query_retrieval_chain = (
    {
        "context": itemgetter("question") | multi_query_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ### Parent Document

# %%
parent_document_retrieval_chain = (
    {
        "context": itemgetter("question") | parent_document_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ### Ensemble

# %%
ensemble_retrieval_chain = (
    {
        "context": itemgetter("question") | ensemble_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ### Semantic Retriever

# %%
semantic_retrieval_chain = (
    {
        "context": itemgetter("question") | semantic_retriever,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# ## Sample Requests

# %%
naive_retrieval_chain.invoke({"question": "What is the most common project domain?"})[
    "response"
].content

# %%
naive_retrieval_chain.invoke({"question": "Were there any usecases about security?"})[
    "response"
].content

# %%
naive_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %%
bm25_retrieval_chain.invoke({"question": "What is the most common project domain?"})[
    "response"
].content

# %%
bm25_retrieval_chain.invoke({"question": "Were there any usecases about security?"})[
    "response"
].content

# %%
bm25_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %%
contextual_compression_retrieval_chain.invoke(
    {"question": "What is the most common project domain?"}
)["response"].content

# %%
contextual_compression_retrieval_chain.invoke(
    {"question": "Were there any usecases about security?"}
)["response"].content

# %%
contextual_compression_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %%
multi_query_retrieval_chain.invoke(
    {"question": "What is the most common project domain?"}
)["response"].content

# %%
multi_query_retrieval_chain.invoke(
    {"question": "Were there any usecases about security?"}
)["response"].content

# %%
multi_query_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %%
parent_document_retrieval_chain.invoke(
    {"question": "What is the most common project domain?"}
)["response"].content

# %%
parent_document_retrieval_chain.invoke(
    {"question": "Were there any usecases about security?"}
)["response"].content

# %%
parent_document_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %%
ensemble_retrieval_chain.invoke(
    {"question": "What is the most common project domain?"}
)["response"].content

# %%
ensemble_retrieval_chain.invoke(
    {"question": "Were there any usecases about security?"}
)["response"].content

# %%
ensemble_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %%
semantic_retrieval_chain.invoke(
    {"question": "What is the most common project domain?"}
)["response"].content

# %%
semantic_retrieval_chain.invoke(
    {"question": "Were there any usecases about security?"}
)["response"].content

# %%
semantic_retrieval_chain.invoke(
    {"question": "What did judges have to say about the fintech projects?"}
)["response"].content

# %% [markdown]
# ### some grouping for requests

# %%
# --- 1) Chain registry (use your existing chain objects) ---
CHAINS = {
    "naive": naive_retrieval_chain,
    "bm25": bm25_retrieval_chain,
    "compression": contextual_compression_retrieval_chain,
    "multi_query": multi_query_retrieval_chain,
    "parent_doc": parent_document_retrieval_chain,
    "ensemble": ensemble_retrieval_chain,
    "semantic": semantic_retrieval_chain,
}


# --- 2) Minimal helpers to normalize outputs ---
def _to_text(resp_dict):
    """Your chains return {'response': <AIMessage|str>, 'context': [...] }."""
    r = resp_dict.get("response")
    if hasattr(r, "content"):  # AIMessage
        return r.content
    return str(r) if r is not None else ""


def _to_context(resp_dict):
    return resp_dict.get("context", [])


# --- 3) Run a single question across selected chains ---
def run_all(question: str, chains=CHAINS):
    results = {}
    for name, ch in chains.items():
        out = ch.invoke({"question": question})
        results[name] = {
            "answer": _to_text(out),
            "contexts": _to_context(out),
        }
    return results


# --- 4) Convenience: quick pretty print for ad-hoc inspection ---
def print_quick(results, max_len=200):
    for name, rec in results.items():
        ans = rec["answer"].strip().replace("\n", " ")
        print(f"[{name}] {ans[:max_len]}{'…' if len(ans) > max_len else ''}")


# %%
# single question across all chains
res = run_all("What is the most common project domain?")
print_quick(res)


# %%
def run_batch(questions, chains=CHAINS):
    """
    Returns: dict[chain_name] -> list of {question, answer, contexts}
    """
    payloads = [{"question": q} for q in questions]
    all_results = {}
    for name, ch in chains.items():
        outs = ch.batch(payloads)
        all_results[name] = [
            {
                "question": q["question"],
                "answer": _to_text(o),
                "contexts": _to_context(o),
            }
            for q, o in zip(payloads, outs)
        ]
    return all_results


def print_results(all_results, max_answer=150, max_context=100, max_ctxs=2):
    """
    Nicely print abridged chain results for inspection.
    """
    for name, records in all_results.items():
        print(f"\n=== {name.upper()} ===")
        for rec in records:
            print(f"Q: {rec['question']}")
            ans = rec["answer"].strip().replace("\n", " ")
            print(f"A: {ans[:max_answer]}{'…' if len(ans) > max_answer else ''}")
            ctxs = rec["contexts"][:max_ctxs]
            for i, c in enumerate(ctxs, 1):
                snippet = c.page_content.strip().replace("\n", " ")
                print(
                    f"  [ctx{i}] {snippet[:max_context]}{'…' if len(snippet) > max_context else ''}"
                )
            print()  # blank line between questions


# %%
QUESTIONS = [
    "What is the most common project domain?",
    "Were there any usecases about security?",
]

batched_results = run_batch(QUESTIONS)
print_results(batched_results)


# %%
