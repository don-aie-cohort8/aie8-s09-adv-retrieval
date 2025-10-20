# %% [markdown]
# # Advanced Retrieval Evaluations - generating RAGAS Golden Testsets

# %%
# Standard library
import getpass
import os
# import openai

# Third-party packages
# LangChain
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
# RAGAS
# from ragas.embeddings import OpenAIEmbeddings # not available in 0.2.10
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    apply_transforms,
)

# %%
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")

# %%
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
# openai_client = openai.OpenAI()

# %% [markdown]
# ## Create the source documents
# 
# - use the LangChain CSVLoader to load the source documents
# - specify the metadata columns that will go into the metadata field
# - any other columns will go into the page_content field

# %% [markdown]
# ### our initial custom approach

# %%
# the effect of this is that we end up with an empty page_content property
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

ragas_usecase_data_attempt_1 = loader.load()

# original code focused on populating the page_content only with the description
# below I'm adding additional columns to improve RAGAS golden testset generation
# QUESTION 1:  if the metadata content duplicates the page_content, is it really metadata?
# QUESTION 2:  what does the structure of the original dataset tell us about the nature of it?  (is it structured or unstructured?)

for doc in ragas_usecase_data_attempt_1:
    title = doc.metadata.get("Project Title", "")
    domain = doc.metadata.get("Project Domain", "")
    secondary = doc.metadata.get("Secondary Domain", "")
    desc = doc.metadata.get("Description", "")

    doc.page_content = f"{title}\nDomain: {domain}\nSecondary Domain: {secondary}\nDescription: {desc}".strip()

# %%
# number of documents
len(ragas_usecase_data_attempt_1)

# %%
# content of the first document
ragas_usecase_data_attempt_1[0]

# %%
# length of the first document's page_content
len(ragas_usecase_data_attempt_1[0].page_content)

# %% [markdown]
# ### revert to more of a standard approach

# %%
# standard approach of using LangChain loaders
# everything that is not specified in metadata_columns goes into the page_content context

ragas_loader = CSVLoader(
    file_path="../data/Projects_with_Domains.csv",
    metadata_columns=[
      "Judge Comments",
      "Score",
      "Project Name",
      "Judge Score"
    ]
)

ragas_usecase_data = ragas_loader.load()

# %%
# number of documents
len(ragas_usecase_data)

# %%
# content of the first document
ragas_usecase_data[0]

# %%
# length of the first document's page_content
len(ragas_usecase_data[0].page_content)

# %% [markdown]
# ## RAGAS Golden Testset Generation

# %% [markdown]
# ### Using the Vanilla Approach

# %%
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
golden_dataset_attempt_1 = generator.generate_with_langchain_docs(ragas_usecase_data, testset_size=10)

# %%
# calculate the length in tokens of the source documents

token_splitter = TokenTextSplitter()

for i, doc in enumerate(ragas_usecase_data):
    tokens = token_splitter._tokenizer.encode(doc.page_content)
    print(f"Doc {i}: {len(tokens)} tokens")

# %% [markdown]
# ### using RAGAS Knowledge Graph functionality
# 
# - unroll the golden testset process to have more control over generation
# - custom personas are not necessary but showcase useful RAGAS functionality

# %% [markdown]
# #### create the graph

# %%
kg = KnowledgeGraph()

for doc in ragas_usecase_data:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

# %%
# check the initial graph

kg

# %%
headline_extractor = HeadlinesExtractor(llm=generator_llm)
headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor
]

apply_transforms(kg, transforms=transforms)

# %%
# check the graph after applying the transforms

kg

# %%
kg.save("usecase_data_kg.json")
usecase_data_kg = KnowledgeGraph.load("usecase_data_kg.json")
usecase_data_kg

# %% [markdown]
# #### identify personas

# %% [markdown]
# | Persona               | Use Case Type                             | Derived From                                                                |
# | --------------------- | ----------------------------------------- | --------------------------------------------------------------------------- |
# | Decision Analyst      | **Asking / Seeking Information**          | “Decision support and information interpretation dominate work-related use” |
# | Domain Researcher     | **Knowledge Graph & Multi-hop Retrieval** | Multi-domain structure in `Projects_with_Domains.csv`                       |
# | Instructional Creator | **Practical Guidance / Tutoring**         | Education & self-learning patterns (10% of usage)                           |
# | AI Practitioner       | **Evaluation & Coding Assistance**        | Work-related “Doing” messages (40% overall)                                 |
# | Creative Strategist   | **Self-Expression / Ideation**            | Growth of “Expressing” and “Creative Guidance” segments                     |
# 

# %%
persona_decision_analyst = Persona(
    name="Decision Analyst",
    role_description=(
        "Uses AI for analytical reasoning and decision support. "
        "Seeks data-driven insights, summaries, and structured outputs to inform business or policy decisions. "
        "Values concise factual responses, traceable evidence, and cost-effective solutions."
    ),
)

persona_domain_researcher = Persona(
    name="Domain Researcher",
    role_description=(
        "Explores multi-domain knowledge sources (e.g., education, health, finance, engineering). "
        "Prefers context-rich retrieval with citations and nuanced synthesis. "
        "Often asks cross-domain 'why/how' questions requiring reasoning beyond surface-level facts."
    ),
)

persona_instructional_creator = Persona(
    name="Instructional Creator",
    role_description=(
        "Designs educational or training materials using AI. "
        "Relies on clear, pedagogical explanations and consistent tone. "
        "Frequently asks for examples, analogies, or simplified explanations for learners."
    ),
)

persona_ai_practitioner = Persona(
    name="AI Practitioner",
    role_description=(
        "Implements and evaluates retrieval-augmented systems. "
        "Needs structured, reproducible outputs like JSON schemas, test cases, and evaluation metrics. "
        "Focuses on precision, recall, and factual grounding when comparing retrievers or datasets."
    ),
)

persona_creative_strategist = Persona(
    name="Creative Strategist",
    role_description=(
        "Uses AI for ideation, storytelling, and persuasive communication. "
        "Seeks novel phrasing, emotional resonance, and creative reframing of ideas. "
        "Frequently explores role-play or scenario-based reasoning."
    ),
)

personas = [
    persona_decision_analyst,
    persona_domain_researcher,
    persona_instructional_creator,
    persona_ai_practitioner,
    persona_creative_strategist,
]

# %% [markdown]
# #### define query behavior

# %%
query_distibution = [
    (
        SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"),
        0.5,
    ),
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name="keyphrases"
        ),
        0.5,
    ),
]

# %% [markdown]
# #### generate testset

# %%
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=usecase_data_kg,
    persona_list=personas,
)

# %%
golden_testset = generator.generate(testset_size=10, query_distribution=query_distibution)
golden_testset.to_pandas()

# %% [markdown]
# ### PLEASE NOTE...
# 
# - in the end the golden testset generation process still creates a graph with no edges
# - the concept of custom personas is still valuable and helps to design better systems

# %% [markdown]
# ### OPTIONAL:   preserve the RAGAS golden testset as jsonl and hugging face datasets
# 
# - demonstrating a few approaches to capture the dataset for later use (including versioning)

# %%
# --- One-Cell Push (AS-IS RAGAS schema, README w/ notebook link, personas in name) ---

import os
import json
import hashlib
import getpass
from datetime import datetime
from typing import List, Dict

from huggingface_hub import HfApi, login, upload_file
from datasets import Dataset, DatasetDict, load_dataset

# =============== USER CONFIG ===============
HF_USERNAME   = "dwb2023"
DATASET_NAME  = "ragas-golden-testset-personas"  # captures personas concept
REPO_ID       = f"{HF_USERNAME}/{DATASET_NAME}"
PRIVATE       = False

LICENSE       = "apache-2.0"
TAGS          = ["ragas", "golden-testset", "rag-eval", "personas"]
LANGS         = ["en"]

# Validate the schema your RAGAS run actually produced (no renaming)
REQUIRED_COLUMNS_AS_IS = {"user_input", "reference", "reference_contexts"}  # add others if present

# Link to the generating notebook
GENERATING_NOTEBOOK_URL = (
    "https://github.com/don-aie-cohort8/aie8-s09-adv-retrieval/blob/main/notebooks/session09-adv-retrieval-ragas.ipynb"
)

PROVENANCE_NOTES = (
    "Generated via RAGAS golden testset pipeline (personas scenario). "
    "See the linked notebook for full pipeline details."
)
# ==========================================

# 0) Login (env var or prompt)
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    token = getpass.getpass("Enter your Hugging Face Hub Token: ")
login(token=token)

# 1) Ensure dataset repo exists
api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

# 2) Convert to HF dataset (AS-IS)
hf_ds = golden_testset.to_hf_dataset()  # Dataset or DatasetDict

# 3) (Optional) JSONL for local inspection
golden_testset.to_jsonl("golden_testset.jsonl")

# 4) Validate AS-IS schema
def _assert_columns(ds, required):
    present = set(ds.column_names)
    missing = required - present
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {sorted(present)}")

if isinstance(hf_ds, DatasetDict):
    for _, ds in hf_ds.items():
        _assert_columns(ds, REQUIRED_COLUMNS_AS_IS)
else:
    _assert_columns(hf_ds, REQUIRED_COLUMNS_AS_IS)

# 5) Build README from real columns + notebook link
def _fingerprint(ds_obj) -> str:
    if isinstance(ds_obj, DatasetDict):
        payload = {k: ds_obj[k].column_names for k in ds_obj.keys()}
        payload["_sizes"] = {k: len(ds_obj[k]) for k in ds_obj.keys()}
    else:
        payload = {"columns": ds_obj.column_names, "rows": len(ds_obj)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

def _schema_list(ds_obj) -> List[str]:
    if isinstance(ds_obj, DatasetDict):
        cols = set()
        for _, ds in ds_obj.items():
            cols |= set(ds.column_names)
        return sorted(cols)
    return sorted(ds_obj.column_names)

def _friendly_descriptions(cols: List[str]) -> Dict[str, str]:
    hints = {
        "user_input": "The generated query/question for evaluation.",
        "reference_contexts": "List of ground-truth passages used as reference context.",
        "reference": "The expected/ground-truth answer.",
        "synthesizer_name": "Name of the synthesizer that produced the sample.",
    }
    return {c: hints.get(c, "") for c in cols}

fp   = _fingerprint(hf_ds)
now  = datetime.utcnow().isoformat() + "Z"
cols = _schema_list(hf_ds)
desc = _friendly_descriptions(cols)

schema_md = "\n".join([f"- **{c}**: {desc[c]}" if desc[c] else f"- **{c}**" for c in cols])
required_md = ", ".join(sorted(REQUIRED_COLUMNS_AS_IS))

readme = f"""---
dataset_info:
  pretty_name: "RAGAS Golden Testset (Personas, AS-IS Schema)"
  task_categories: ["question-answering"]
  license: "{LICENSE}"
  language: {json.dumps(LANGS)}
  tags: {json.dumps(TAGS)}
---

# RAGAS Golden Testset — Personas (AS-IS)

- **Generated**: {now}
- **Repo**: `{REPO_ID}`
- **Fingerprint**: `{fp}`
- **Provenance**: {PROVENANCE_NOTES}
- **Generating Notebook**: {GENERATING_NOTEBOOK_URL}

## Schema (as produced by RAGAS)
{schema_md}

**Required columns validated in notebook:** {required_md}

> Note: `reference_contexts` is intentionally kept as a list to reflect the native RAGAS output for teaching.
"""

upload_file(
    path_or_fileobj=readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)

# 6) Push and verify
print(f"Pushing to {REPO_ID} (private={PRIVATE}) ...")
hf_ds.push_to_hub(REPO_ID, private=PRIVATE)
print("Push complete. Verifying...")
loaded = load_dataset(REPO_ID)
print(loaded)
print("✅ Done (AS-IS schema, personas captured in dataset name).")


# %%
# ragas_usecase_data

# %%
# Convert the list of documents into a dictionary compatible with the `datasets` library
data_dict = {
    "page_content": [doc.page_content for doc in ragas_usecase_data],
    "metadata": [doc.metadata for doc in ragas_usecase_data]
}


# %%
hf_dataset = Dataset.from_dict(data_dict)

# Print the dataset to confirm the structure
print(hf_dataset)
# Output:
# Dataset({
#     features: ['page_content', 'metadata'],
#     num_rows: 2
# })


# %%
# push the ragas_usecase_data dataset to hugging face

hf_dataset.push_to_hub("dwb2023/ragas-usecase-raw-data")


# %%



