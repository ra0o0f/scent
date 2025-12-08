

[Full technical documentation](scent-technical-doc.pdf)

# Scent: Constrained Generation Framework

Scent is a constrained generation framework for entity-linking that grounds textual output in structured graph knowledge. Scent utilizes an encoder backbone (e.g., RoBERTa) and introduces two specialized, simultaneously trained Low-Rank Adaptation (LoRA) modules:

- **Graph-adapter:** Designed to produce semantic representations of entities derived from structured information.
- **Text-adapter:** Manages the integration of textual context with structure-grounded entity slots.

## Framework Overview

The generation is **constrained**: the output must be selected from a pre-defined candidate set of entities, ensuring the text is grounded in the Knowledge Graph (KG).

### Dual-View Representation
The target entity is processed through two distinct but aligned views:

1.  **The Textual View:** Presents the target entity within its linguistic context. We prepare the sequence for prediction by defining the entity slot with entity boundary markers and a fixed buffer of mask tokens.
2.  **The Structural View:** Linearizes its local neighborhood (k-hop traversal) into a sequence of node and edge semantic units. We generate semantic representations directly from the text.

### Training Objectives
Scent employs a multi-task strategy to enforce consistency between the textual output and the underlying graph topology:

* **Textual Generation:** Using the **Text-Adapter**, we perform standard Masked Language Modeling on the tokens within the mask buffer.
* **Structure-Aware Feature Regression:** Using the **Graph-Adapter**, we apply a Masked Feature Regression on the sequence. The model must reconstruct the semantic vector of the masked target node solely from its structural context.
* **Cross-Modal Alignment:** To bridge the two views, we enforce a strict consistency constraint on the `<node_start>` token.
* **Representation Regularization:** We apply a decorrelation objective to the batch of predicted graph-view representations to penalize redundancy between feature dimensions.

## Inference and Constrained Generation

Scent introduces a high-efficiency inference mechanism designed to circumvent latency bottlenecks. We score all potential entities from the candidate set simultaneously in a single forward pass.

### 1. Offline Structural Indexing
Prior to inference, we construct a dense vector index of the Knowledge Graph using the Graph-Adapter. This map provides the "ground truth" semantic vectors against which text generation will be reranked.

### 2. Parallel Linguistic Evaluation
Scent evaluates the likelihood of the entire candidate vocabulary across all mask positions in parallel. We employ a highly optimized tensor "gather" operation to extract specific log-probabilities corresponding to the pre-tokenized sequences of the candidate set.

### 3. Cross-Modal Alignment and Re-ranking
Scent employs a **cascade re-ranking strategy**. We utilize the linguistic scores as a high-recall filter to prune the search space. The structural verification is computed exclusively for this reduced subset. We calculate the final score as a weighted combination of the linguistic probability and the cosine similarity between the predicted and actual graph vectors.


```mermaid
flowchart TD
    classDef purple fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#2e1065
    classDef blue fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a
    classDef dark fill:#1e1b4b,stroke:#312e81,stroke-width:2px,color:#fff

    subgraph Inputs
        direction TB
        style Inputs fill:transparent,stroke:none
        IT[Text View Input<br/>Masked Sequence]:::purple
        IG[Structural View<br/>Linearized Graph]:::purple
        IS[Isolated Node<br/>Sequence]:::purple
    end

    subgraph SharedBackbone [Shared Encoder Backbone]
        direction TB
        style SharedBackbone fill:#eff6ff,stroke:#bfdbfe,stroke-dasharray: 5 5
        TA[Text-Adapter]:::blue
        GA[Graph-Adapter]:::blue
    end

    IT --> TA
    IG --> GA
    IS --> GA

    MLM[MLM Head]:::dark
    TA --> MLM
    MLM --> L1((Cross Entropy)):::purple

    TA --> L3((Distance)):::purple
    GA --> L3

    GA --> L2((Feature Regression)):::purple
    GA --> L4((Decorrelation)):::purple
```

# Datasets

To replicate the experiments or run the system, you need to download and prepare the following datasets.

### 1. Entity Linking (AIDA-YAGO2)
We use the **AIDA-YAGO2** dataset for entity linking training and preparing the candidate sets.

* **Download:** You can download the dataset from the [official AIDA resource page](https://resources.mpi-inf.mpg.de/yago-naga/aida/downloads.html).
* **Preparation:** Please note that the raw download requires processing. Follow the instructions provided within the downloaded files to generate the consolidated TSV file containing both training and test sets.
    * **Target Output File:** `AIDA-YAGO2-dataset.tsv`

### 2. Graph Database (YAGO4)
The underlying graph database is built using **YAGO4**. You will need to download specific files from the **English Wikipedia** section of the [YAGO4 downloads](https://yago-knowledge.org/downloads/yago-4).

We require the following two components:

1.  **Facts:** Facts that are not labels.
2.  **Labels:** All entity labels (including `rdfs:label`, `rdfs:comment`, and `schema:alternateName`).

Please ensure you download the following specific files:

| Component | Filename |
| :--- | :--- |
| **Facts** | `yago-wd-facts.nt.gz` |
| **Labels** | `yago-wd-labels.nt.gz` |

---

### Graph Data Preparation
Once the datasets are downloaded, you must prepare the graph data for training and inference. This process involves parsing the raw YAGO files and structuring them for the efficient reading.

To run the preprocessing pipeline, execute the following command:

```bash
python -m app.main --mode preprocess
```

The script relies on file paths defined in the configuration. The default configuration file is located at **configs/config.yaml**. You can modify this file to point to your specific download locations.

