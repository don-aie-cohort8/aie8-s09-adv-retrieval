# %% [markdown]
# # Advanced Retrieval with LangChain
# 
# In the following notebook, we'll explore various methods of advanced retrieval using LangChain!
# 
# We'll touch on:
# 
# - Naive Retrieval
# - Best-Matching 25 (BM25)
# - Multi-Query Retrieval
# - Parent-Document Retrieval
# - Contextual Compression (a.k.a. Rerank)
# - Ensemble Retrieval
# - Semantic chunking
# 
# We'll also discuss how these methods impact performance on our set of documents with a simple RAG chain.
# 
# There will be two breakout rooms:
# 
# - 🤝 Breakout Room Part #1
#   - Task 1: Getting Dependencies!
#   - Task 2: Data Collection and Preparation
#   - Task 3: Setting Up QDrant!
#   - Task 4-10: Retrieval Strategies
# - 🤝 Breakout Room Part #2
#   - Activity: Evaluate with Ragas

# %% [markdown]
# # 🤝 Breakout Room Part #1

# %% [markdown]
# ## Task 1: Getting Dependencies!
# 
# We're going to need a few specific LangChain community packages, like OpenAI (for our [LLM](https://platform.openai.com/docs/models) and [Embedding Model](https://platform.openai.com/docs/guides/embeddings)) and Cohere (for our [Reranker](https://cohere.com/rerank)).

# %% [markdown]
# We'll also provide our OpenAI key, as well as our Cohere API key.

# %%
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")

# %%
os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")

# %% [markdown]
# ## Task 2: Data Collection and Preparation
# 
# We'll be using our Use Case Data once again - this time the strutured data available through the CSV!

# %% [markdown]
# ### Data Preparation
# 
# We want to make sure all our documents have the relevant metadata for the various retrieval strategies we're going to be applying today.

# %%
from langchain_community.document_loaders.csv_loader import CSVLoader
from datetime import datetime, timedelta

loader = CSVLoader(
    file_path=f"./data/Projects_with_Domains.csv",
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

for doc in synthetic_usecase_data:
    doc.page_content = doc.metadata["Description"]

# %% [markdown]
# Let's look at an example document to see if everything worked as expected!

# %%
synthetic_usecase_data[0]

# %% [markdown]
# ## Task 3: Setting up QDrant!
# 
# Now that we have our documents, let's create a QDrant VectorStore with the collection name "Synthetic_Usecases".
# 
# We'll leverage OpenAI's [`text-embedding-3-small`](https://openai.com/blog/new-embedding-models-and-api-updates) because it's a very powerful (and low-cost) embedding model.
# 
# > NOTE: We'll be creating additional vectorstores where necessary, but this pattern is still extremely useful.

# %%
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Qdrant.from_documents(
    synthetic_usecase_data,
    embeddings,
    location=":memory:",
    collection_name="Synthetic_Usecases"
)

# %% [markdown]
# ## Task 4: Naive RAG Chain
# 
# Since we're focusing on the "R" in RAG today - we'll create our Retriever first.

# %% [markdown]
# ### R - Retrieval
# 
# This naive retriever will simply look at each review as a document, and use cosine-similarity to fetch the 10 most relevant documents.
# 
# > NOTE: We're choosing `10` as our `k` here to provide enough documents for our reranking process later

# %%
naive_retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

# %% [markdown]
# ### A - Augmented
# 
# We're going to go with a standard prompt for our simple RAG chain today! Nothing fancy here, we want this to mostly be about the Retrieval process.

# %%
from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# %% [markdown]
# ### G - Generation
# 
# We're going to leverage `gpt-4.1-nano` as our LLM today, as - again - we want this to largely be about the Retrieval process.

# %%
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4.1-nano")

# %% [markdown]
# ### LCEL RAG Chain
# 
# We're going to use LCEL to construct our chain.
# 
# > NOTE: This chain will be exactly the same across the various examples with the exception of our Retriever!

# %%
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

naive_retrieval_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# Let's see how this simple chain does on a few different prompts.
# 
# > NOTE: You might think that we've cherry picked prompts that showcase the individual skill of each of the retrieval strategies - you'd be correct!

# %%
naive_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
naive_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
naive_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# Overall, this is not bad! Let's see if we can make it better!

# %% [markdown]
# ## Task 5: Best-Matching 25 (BM25) Retriever
# 
# Taking a step back in time - [BM25](https://www.nowpublishers.com/article/Details/INR-019) is based on [Bag-Of-Words](https://en.wikipedia.org/wiki/Bag-of-words_model) which is a sparse representation of text.
# 
# In essence, it's a way to compare how similar two pieces of text are based on the words they both contain.
# 
# This retriever is very straightforward to set-up! Let's see it happen down below!
# 

# %%
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(synthetic_usecase_data)

# %% [markdown]
# We'll construct the same chain - only changing the retriever.

# %%
bm25_retrieval_chain = (
    {"context": itemgetter("question") | bm25_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# Let's look at the responses!

# %%
bm25_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
bm25_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
bm25_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# It's not clear that this is better or worse, if only we had a way to test this (SPOILERS: We do, the second half of the notebook will cover this)

# %% [markdown]
# #### ❓ Question #1:
# 
# Give an example query where BM25 is better than embeddings and justify your answer.
# 
# ##### ✅ Answer
# 

# %% [markdown]
# ## Task 6: Contextual Compression (Using Reranking)
# 
# Contextual Compression is a fairly straightforward idea: We want to "compress" our retrieved context into just the most useful bits.
# 
# There are a few ways we can achieve this - but we're going to look at a specific example called reranking.
# 
# The basic idea here is this:
# 
# - We retrieve lots of documents that are very likely related to our query vector
# - We "compress" those documents into a smaller set of *more* related documents using a reranking algorithm.
# 
# We'll be leveraging Cohere's Rerank model for our reranker today!
# 
# All we need to do is the following:
# 
# - Create a basic retriever
# - Create a compressor (reranker, in this case)
# 
# That's it!
# 
# Let's see it in the code below!

# %%
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=naive_retriever
)

# %% [markdown]
# Let's create our chain again, and see how this does!

# %%
contextual_compression_retrieval_chain = (
    {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %%
contextual_compression_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
contextual_compression_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
contextual_compression_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# We'll need to rely on something like Ragas to help us get a better sense of how this is performing overall - but it "feels" better!

# %% [markdown]
# ## Task 7: Multi-Query Retriever
# 
# Typically in RAG we have a single query - the one provided by the user.
# 
# What if we had....more than one query!
# 
# In essence, a Multi-Query Retriever works by:
# 
# 1. Taking the original user query and creating `n` number of new user queries using an LLM.
# 2. Retrieving documents for each query.
# 3. Using all unique retrieved documents as context
# 
# So, how is it to set-up? Not bad! Let's see it down below!
# 
# 

# %%
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever, llm=chat_model
) 

# %%
multi_query_retrieval_chain = (
    {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %%
multi_query_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
multi_query_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
multi_query_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# #### ❓ Question #2:
# 
# Explain how generating multiple reformulations of a user query can improve recall.
# 
# ##### ✅ Answer
# 

# %% [markdown]
# ## Task 8: Parent Document Retriever
# 
# A "small-to-big" strategy - the Parent Document Retriever works based on a simple strategy:
# 
# 1. Each un-split "document" will be designated as a "parent document" (You could use larger chunks of document as well, but our data format allows us to consider the overall document as the parent chunk)
# 2. Store those "parent documents" in a memory store (not a VectorStore)
# 3. We will chunk each of those documents into smaller documents, and associate them with their respective parents, and store those in a VectorStore. We'll call those "child chunks".
# 4. When we query our Retriever, we will do a similarity search comparing our query vector to the "child chunks".
# 5. Instead of returning the "child chunks", we'll return their associated "parent chunks".
# 
# Okay, maybe that was a few steps - but the basic idea is this:
# 
# - Search for small documents
# - Return big documents
# 
# The intuition is that we're likely to find the most relevant information by limiting the amount of semantic information that is encoded in each embedding vector - but we're likely to miss relevant surrounding context if we only use that information.
# 
# Let's start by creating our "parent documents" and defining a `RecursiveCharacterTextSplitter`.

# %%
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

parent_docs = synthetic_usecase_data
child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

# %% [markdown]
# We'll need to set up a new QDrant vectorstore - and we'll use another useful pattern to do so!
# 
# > NOTE: We are manually defining our embedding dimension, you'll need to change this if you're using a different embedding model.

# %%
from langchain_qdrant import QdrantVectorStore

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

parent_document_vectorstore = QdrantVectorStore(
    collection_name="full_documents", embedding=OpenAIEmbeddings(model="text-embedding-3-small"), client=client
)

# %% [markdown]
# Now we can create our `InMemoryStore` that will hold our "parent documents" - and build our retriever!

# %%
store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore = parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# %% [markdown]
# By default, this is empty as we haven't added any documents - let's add some now!

# %%
parent_document_retriever.add_documents(parent_docs, ids=None)

# %% [markdown]
# We'll create the same chain we did before - but substitute our new `parent_document_retriever`.

# %%
parent_document_retrieval_chain = (
    {"context": itemgetter("question") | parent_document_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# Let's give it a whirl!

# %%
parent_document_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
parent_document_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
parent_document_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# Overall, the performance *seems* largely the same. We can leverage a tool like [Ragas]() to more effectively answer the question about the performance.

# %% [markdown]
# ## Task 9: Ensemble Retriever
# 
# In brief, an Ensemble Retriever simply takes 2, or more, retrievers and combines their retrieved documents based on a rank-fusion algorithm.
# 
# In this case - we're using the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) algorithm.
# 
# Setting it up is as easy as providing a list of our desired retrievers - and the weights for each retriever.

# %%
from langchain.retrievers import EnsembleRetriever

retriever_list = [bm25_retriever, naive_retriever, parent_document_retriever, compression_retriever, multi_query_retriever]
equal_weighting = [1/len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list, weights=equal_weighting
)

# %% [markdown]
# We'll pack *all* of these retrievers together in an ensemble.

# %%
ensemble_retrieval_chain = (
    {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# Let's look at our results!

# %%
ensemble_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
ensemble_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
ensemble_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# ## Task 10: Semantic Chunking
# 
# While this is not a retrieval method - it *is* an effective way of increasing retrieval performance on corpora that have clean semantic breaks in them.
# 
# Essentially, Semantic Chunking is implemented by:
# 
# 1. Embedding all sentences in the corpus.
# 2. Combining or splitting sequences of sentences based on their semantic similarity based on a number of [possible thresholding methods](https://python.langchain.com/docs/how_to/semantic-chunker/):
#   - `percentile`
#   - `standard_deviation`
#   - `interquartile`
#   - `gradient`
# 3. Each sequence of related sentences is kept as a document!
# 
# Let's see how to implement this!

# %% [markdown]
# We'll use the `percentile` thresholding method for this example which will:
# 
# Calculate all distances between sentences, and then break apart sequences of setences that exceed a given percentile among all distances.

# %%
from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# %% [markdown]
# Now we can split our documents.

# %%
semantic_documents = semantic_chunker.split_documents(synthetic_usecase_data[:20])

# %% [markdown]
# Let's create a new vector store.

# %%
semantic_vectorstore = Qdrant.from_documents(
    semantic_documents,
    embeddings,
    location=":memory:",
    collection_name="Synthetic_Usecase_Data_Semantic_Chunks"
)

# %% [markdown]
# We'll use naive retrieval for this example.

# %%
semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k" : 10})

# %% [markdown]
# Finally we can create our classic chain!

# %%
semantic_retrieval_chain = (
    {"context": itemgetter("question") | semantic_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# %% [markdown]
# And view the results!

# %%
semantic_retrieval_chain.invoke({"question" : "What is the most common project domain?"})["response"].content

# %%
semantic_retrieval_chain.invoke({"question" : "Were there any usecases about security?"})["response"].content

# %%
semantic_retrieval_chain.invoke({"question" : "What did judges have to say about the fintech projects?"})["response"].content

# %% [markdown]
# #### ❓ Question #3:
# 
# If sentences are short and highly repetitive (e.g., FAQs), how might semantic chunking behave, and how would you adjust the algorithm?
# 
# ##### ✅ Answer
# 

# %% [markdown]
# # 🤝 Breakout Room Part #2

# %% [markdown]
# #### 🏗️ Activity #1
# 
# Your task is to evaluate the various Retriever methods against eachother.
# 
# You are expected to:
# 
# 1. Create a "golden dataset"
#  - Use Synthetic Data Generation (powered by Ragas, or otherwise) to create this dataset
# 2. Evaluate each retriever with *retriever specific* Ragas metrics
#  - Semantic Chunking is not considered a retriever method and will not be required for marks, but you may find it useful to do a "semantic chunking on" vs. "semantic chunking off" comparision between them
# 3. Compile these in a list and write a small paragraph about which is best for this particular data and why.
# 
# Your analysis should factor in:
#   - Cost
#   - Latency
#   - Performance
# 
# > NOTE: This is **NOT** required to be completed in class. Please spend time in your breakout rooms creating a plan before moving on to writing code.

# %% [markdown]
# ##### HINTS:
# 
# - LangSmith provides detailed information about latency and cost.

# %%
### YOUR CODE HERE


