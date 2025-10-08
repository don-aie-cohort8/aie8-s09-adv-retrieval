# Assessing Advanced Retrievers with RAGAS

- Now that we have all of the advanced retrieval methods we need to evaluate them against each other using RAGAS metrics to see which ones are worth it for our use case.

## Objectives

1. Create a "Golden Dataset"
    - Use SDG to create a dataset for evaluation.
2. Evaluate Each Retriever Using RAGAS Metrics
    - Compare different retrievers with retriever-specific evaluation metrics.
    - Note: Semantic Chunking is NOT a retriever, but you can test its impact by comparing "semantic chunking on" vs. "semantic chunking off"
3. Compile Findings & Write a Short Analysis
    - Determine which retriever performs best for this dataset and explain why.
    - Your analysis should factor in:
        - Cost (How expensive is it to run?)
        - Latency (How fast is retrieval?)
        - Performance (How well does it retrieve the right data?)

---

## Plan and Start working on Eval

- collaborate and map out their approach before coding.
- Key considerations
  - What trade-offs are you willing to accept?
  - Is faster retrieval always better, or does accuracy matter more?

---

## Takeaways

- creating and implementing based on a plan is essential to the success of your work for this session.
- collaborate asynchronously in Discord or within their journey groups.
- findings should be compiled into a short paragraph explaining their choice.
