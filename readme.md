

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

-----

## Datasets

To replicate the experiments or run the system, we employ the following resources for graph construction, training, and evaluation.

### 1\. Knowledge Graph (YAGO4)

We construct our graph structure using **YAGO4**, leveraging its rich taxonomy and rigorous type constraints.

  * **Source:** [YAGO4 Downloads (English Wikipedia)](https://yago-knowledge.org/downloads/yago-4)
  * **Required Files:** We specifically require the following components to separate facts from labels:

| Component | Filename | Description |
| :--- | :--- | :--- |
| **Facts** | `yago-wd-facts.nt.gz` | Facts that are not labels. |
| **Labels** | `yago-wd-labels.nt.gz` | All entity labels (including `rdfs:label`, `rdfs:comment`, and `schema:alternateName`). |

### 2\. Entity Abstracts (Structured Wikipedia)

To generate the semantic initialization for graph nodes (via the "First-Last-Avg" strategy), we extract short abstracts for every entity from the **Structured Wikipedia** dataset.

  * **Source:** [HuggingFace - Structured Wikipedia](https://huggingface.co/datasets/wikimedia/structured-wikipedia)

### 3\. Training Data (AIDA-YAGO2)

We train the model exclusively on the training split of the **AIDA-YAGO2** dataset. For Scent, mentions in AIDA-YAGO2 have been mapped to their corresponding nodes in YAGO4 to create ground truth targets.

  * **Source:** [AIDA-YAGO2 Downloads](https://resources.mpi-inf.mpg.de/yago-naga/aida/downloads.html)
  * **Preparation:** Process the raw download to generate the consolidated TSV file (`AIDA-YAGO2-dataset.tsv`).

### 4\. Robustness Evaluation (ShadowLink)

To evaluate the model's resilience against prior bias, we utilize the **ShadowLink** benchmark. This dataset is specifically designed to assess "Entity Overshadowing", a phenomenon where models favor common entities (Top) over less popular ones (Shadow) that share the same surface form.

  * **Source:** [HuggingFace - ShadowLink](https://huggingface.co/datasets/vera-pro/ShadowLink)

-----

## Usage and Execution

The system entry point handles preprocessing, training, and evaluation (including standard AIDA evaluation and ShadowLink robustness checks).

### 1\. Preprocessing

Before training, parse the raw dataset files and structure the data for efficient reading.

```bash
python main.py --mode preprocess --config configs/config.yaml
```

### 2\. Training

To train the Scent model:

```bash
python main.py --mode train --graph-training True
```

### 3\. Evaluation

The system supports two evaluation modes. You must specify the checkpoint path to load the trained model.

**Standard Evaluation (AIDA-YAGO2):**

```bash
python main.py --mode evaluate --eval-mode aida --eval-checkpoint path/to/model.pt
```

**Robustness Evaluation (ShadowLink):**

```bash
python main.py --mode evaluate --eval-mode shadowlink --eval-checkpoint path/to/model.pt
```

### Configuration

All file paths (datasets, graph files) and model hyperparameters are defined in the configuration file (`configs/config.yaml`). Ensure this file is updated to point to the specific download locations of the datasets listed above.