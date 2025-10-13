# **Session 09: SDG & LangGraph – Code Walkthrough Summary**

**Date:** 2025-10-13    **Presenter:** Don B
**Repo:** [`aie8-s09-adv-retrieval`](https://github.com/don-aie-cohort8/aie8-s09-adv-retrieval)

---

## 🧠 Overview

This session reviewed major lessons from **Sessions 7-9** and walked through code refactors improving the **RAGAS + LangGraph** workflow.
Focus areas included dataset reliability, retriever design, version migration (0.2 → 0.3), and observability.

---

## 🔁 RAGAS Version Migration

| Area             | v0.2 Behavior           | v0.3 Change                                                      |
| ---------------- | ----------------------- | ---------------------------------------------------------------- |
| Embeddings       | Uses LangChain wrappers | Direct OpenAI/Other API calls (expected deprecation of wrappers) |
| Graph Generation | Stable for single-hop   | Multi-hop still unstable — start simple                          |
| Persona Handling | Limited and text-based  | More robust generation via structured prompts                    |

**Key Tip:** Begin with **single-hop** data generation before multi-hop to avoid silent failures.

---

## 🧩 Dataset and Tokenization Findings

* The *“projects with domains”* dataset is **semi-structured** and too small → frequent
  `_Documents appear to be too short (< 100 tokens)_` errors.
* **Duplication & mismatched labels** (“sales CRM” tagged as healthcare) → unreliable semantic retrieval.
* **Fix:** Concatenate `domain` + `secondary_domain` + `description` into `page_content`.
* Validate length via `TokenTextSplitter`; target ≥ 100 tokens per doc.

---

## 🧱 LangChain Document Loader Behavior

* All declared metadata fields → `metadata` dict.
* All other fields → `page_content`.
* Earlier notebooks stored everything in metadata (“kitchen sink”).
  ✅ **Refactor:** Keep meaningful text in `page_content` for retrieval fidelity.

---

## 🧍‍♂️ Persona-Driven Test Set Generation

* Implemented **custom personas** (e.g., ChatGPT user archetypes) to drive query diversity.
* Compared v0.2 vs v0.3 behavior: 0.3 produces richer synthetic data and more queries per persona.
* Personas plugged directly into the generator → observable impact on query distributions.

---

## 🧮 Retriever & Vector Store Architecture

* **Vector Stores:** Semantic (Qdrant), Parent/Child, and in-memory.
* **Retrievers:** Compression, Multi-Query, and Ensemble.
* All retrievers grouped under a unified API interface → ready for **FastAPI/FastMCP** exposure.
* Mirrors real-world RAG pattern: parent context chunks for coarse recall + fine semantic retrieval.

---

## 📊 Evaluation & Observability

* Helpers normalize outputs and run single/batch queries across retrievers.
* **Duplicates** stem from CSV data, not retriever logic.
* Validation via **LangSmith traces** and token-level inspection ensures pipeline truthfulness.
* Demonstrates how to extend these retrievers for RAGAS metric evaluation and LangSmith reporting.

---

## 🌐 Publishing & Provenance

* Pushed source and generated datasets to **Hugging Face (`dwb2023`)** for transparency and reproducibility.
* Each dataset includes metadata descriptions and link back to the original class notebook
  → see [`session09-adv-retrieval-ragas.ipynb`](https://github.com/don-aie-cohort8/aie8-s09-adv-retrieval/blob/main/notebooks/session09-adv-retrieval-ragas.ipynb)

---

## ✅ Key Takeaways

1. Start simple (single-hop) → multi-hop later.
2. Preserve semantic content in `page_content`; avoid metadata overload.
3. Maintain ≥ 100 tokens per doc to prevent RAGAS rejections.
4. Use personas to simulate authentic user queries.
5. Expose retrievers via MCP / FastAPI for LLM integration.
6. Track runs and validate via LangSmith & Hugging Face datasets.
