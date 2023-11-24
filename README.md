# Advanced RAG Queries Using EvaDB

Retrieval-Augmented Generation (RAG) has been proved powerful in answer context-specific user queries. In this project, we re-implemented serveral RAG modules in Llamaindex(https://www.llamaindex.ai/) using EvaDB(https://evadb.readthedocs.io/en/stable/index.html).

## Installation & Example

To install the dependencies and run the examples, run
```
pip install -r requirements.txt
```

Below is an example query:

> ⏳ Connecting to EvaDB...
> ✅ Connected to EvaDB!
> Please provide your OpenAI API key (will be hidden): 
> Please enter your question: Which city mentioned has the highest population?
> 🤑 LLM call cost: $0.0002
> context retrieval time: 40.811262130737305
> 🤑 LLM call cost: $0.0062
> response synthesis time: 0.8700895309448242
> answer: Toronto has the highest population mentioned, with a recorded population of 2,794,356 in 2021. This is higher than the populations of Boston and Atlanta mentioned in the context.
> cost: 0.00622

## Overview

Llamaindex adopts RAG-based frameworks to serve user queries. While these RAG-Based frameworks can be sophisticated and vary greatly in structure, all of them contain three basic components:

**Data Warehouse:** A collection of source data that provides context information for the user query.

**Retriever:** Since the data warehouse is typically too large to fit entirely in one LLM window, and because usually only a small subset of data is useful for a given query, we need a `Retriever` to retrieve the most relevant context information.

**Response Synthesizer:** This is the part responsible for directly interacting with the LLM and using it to generate an answer to the query given the context information.

## Implementation Details

We implemented 3 different `Retriever`s and 3 `ResponseSynthesizer`s, plus 2 `QueryEngine`s. We give a sketch of how each of them works.

### Retrievers

**Summary Index Retriever:** This is the simplest retriever; it retrieves the entire data warehouse as context information. This retriever is typically used for summary tasks (for example, concluding the author's overall attitude), as these tasks often require information from all source data.

**Vector Index Retriever:** Source data is stored as separate chunks, each of whose semantic is captured using an embedding vector. On retrieval, the retriever first embeds the user query, and then searches for the most relevant context using a vector index (we use `FAISS` in our implementation).

**Keyword Table Index Retriever:** Each source data chunk is summarized by a few keywords extracted from it. On retrieval, the retriever first extracts keywords from the user query, and then searches for the most relevant context chunks in terms of the number of keyword matches. Keyword matching can be done using either exact match or semantic match.

### Response Synthesizers

**Compact Response Synthesizer:** This is the simplest form of response synthesizer; it gives all context chunks to the LLM in one go and gets the answer from the LLM.

**Refine Response Synthesizer:** Source data chunks are grouped in batches of $K$. The response synthesizer gives context information to the LLM one batch at a time. In the first iteration, the LLM is asked to generate an answer using the first data batch. In subsequent iterations, the LLM is given one more batch and the previous answer, and is asked to refine the previous answer based on the additional context.

**Tree Summarize Response Synthesizer:** This response synthesizer summarizes the context information in a "tree" manner. $N$ source data chunks are given to the LLM in batches of $K$, and the LLM is asked to answer the question based on each of these batches, generating about $\frac{N}{K}$ answers. In subsequent iterations, the LLM uses the answers from the last iterations as context information in batches of $K$, and produces answers based on them. The algorithm terminates when the final answer is produced, in about $\log_K N$ rounds.

### Query Engines

**Simple Query Engine:** This is the most straightforward query engine, consisting of one retriever and one response synthesizer. On serving a user query, it retrieves context information using the retriever, and generates the response using the response synthesizer.

**Retry Query Engine:** This query engine incorporates an LLM-based `Evaluator`. When an answer is produced, instead of returning it directly to the user, it first grades it using the evaluator. If the generated answer fails, the query engine re-executes the query. The above is repeated until the answer passes or a max number of retries is reached.