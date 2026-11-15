# Agent Smith: Relevant-History Selection for Safe Tool-Using Language Agents

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![PEFT / LoRA](https://img.shields.io/badge/PEFT-LoRA%20r%3D8-orange.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LMA Group Project Final Report Codebase**  
**Authors:** Emil Joji, Raj K Jain, Shobhan Parida, Yashwanth B K  
**Date:** November 2026  

---

## Overview

When autonomous language model agents interact with external environments (e.g., local filesystems, cloud infrastructure, databases, and financial transaction APIs), robust pre-execution safety verification is crucial. In real-world enterprise operations, safety policies are inherently **state-dependent**: whether an action is permissible depends strictly on earlier events in the interaction trajectory (e.g., explicit user authorizations, revoked tokens, scope lockouts, or compliance constraints).

As agent workflows unfold over tens or hundreds of operations, vital policy events are inevitably displaced from the active context window by intervening tool outputs, telemetry logs, and intermediate reasoning steps. Standard language agent runtimes employ a recency-biased sliding window that exposes only the latest $k$ interactions. Under this regime, the safety monitor lacks the prerequisite evidence to distinguish between authorized executions and catastrophic security violations, resulting in severe vulnerability to position-induced displacement and untrusted tool manipulation.

**Agent Smith** investigates whether a specialized, lightweight safety monitor achieves superior pre-execution classification fidelity when supplied with **policy-relevant historical interactions** retrieved from the full trajectory under a fixed token budget ($B = 256$ history tokens, 512 total context tokens), rather than relying on recency or unbounded context expansion.

> *"We measure both evidence recovery and downstream verdicts because a system can recover decisive evidence and still make the wrong decision."*

---

## Key Findings

| Metric / Evaluation Dimension | Recency (H1) | BM25 Sparse (H2) | Learned Rerank (H3) | Hybrid (H3 + Policy Checker) |
| :--- | :---: | :---: | :---: | :---: |
| **Decisive Evidence Recovery** (Held-Out Test / 288) | 69 (24.0%) | 280 (97.2%) | **288 (100.0%)** | **288 (100.0%)** |
| **Pre-Packing nDCG@5** | — | 0.897 | **0.997–0.998** | **0.997–0.998** |
| **Neural Verdict Accuracy** (Seed 17 / 360) | 105 (29.2%) | 206 (57.2%) | 201 (55.8%) | — |
| **Neural Verdict Accuracy** (Seed 42 / 360) | 113 (31.4%) | 211 (58.6%) | 216 (60.0%) | — |
| **Hybrid Verified Accuracy** (Seeds 17 & 42 / 360) | 141 (39.2%) | 352 (97.8%) | — | **360 (100.0%)** |
| **Legitimate Proposals Allowed** (/ 144) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **144 (100.0%)** |
| **Prohibited Actions Blocked** (Seed 42 / 144) | 43 (29.9%) | 144 (100.0%) | 144 (100.0%) | **144 (100.0%)** |
| **Agentic Workflow Correctness** (/ 8 fixtures) | — | — | 3/8 (Qwen) | **8/8 (Fixed + Investigator)** |

### Core Empirical Insights:
1. **Recency Catastrophically Fails:** Sliding-window recency (H1) misses crucial historical evidence in 76.0% of answerable cases, consistently leading to unsafe execution or incorrect deferrals.
2. **Learned Reranking Reaches Gold Evidence Recovery:** Fine-tuned cross-encoder reranking (H3) recovers complete decisive evidence in **288/288 (100.0%)** answerable cases across six held-out enterprise domains, outperforming BM25 (280/288).
3. **Trajectory Length Invariance:** H3 accuracy was strictly invariant across trajectory lengths ($T \in \{8, 16, 32\}$): scoring exactly **67/120** per length in Seed 17, and **72/120** per length in Seed 42.
4. **Small Neural Monitor Pathology:** While the interim pilot suffered from 0% `block` recall (over-permissive gatekeeper), the multi-seed model trained on out-of-fold contexts completely inverted to **0% `allow` recall** (hyper-conservative paralysis, 0 allows issued on 144 legitimate test cases per seed).
5. **The Proposal Validation Bottleneck:** In autonomous agent trials with frozen `Qwen2.5-1.5B-Instruct`, 7/8 candidate tool proposals failed strict JSON controller schema validation, identifying schema compliance as a primary operational barrier.
6. **Hybrid Breakthrough (100% Accuracy):** Decoupling learned neural retrieval (H3) from policy adjudication via a deterministic policy checker achieved **360/360 (100.0%)** held-out accuracy and **8/8 verified executions** on investigator agent workflows.

---

## System Architecture

Agent Smith enforces a 5-stage complete mediation pipeline prior to any environmental state mutation:

```
[ Worker Agent (Qwen2.5) ]
            │ (emits candidate proposal p)
            ▼
[ Stage 1: Controller Gate ] ────(invalid JSON / unauthorized)────► [ Halt & Reject ]
            │ (syntactic & schema pass)
            ▼
[ Stage 2: History Selector ] ───(queries full history H)────────► [ Rank & Filter ]
            │                                                               │
            │ (causal temporal packing: resource, scope, version, attr)     ▼
            ▼                                                      [ Token Budget B=256 ]
[ Stage 3: Context Packing ] ◄─────────────────────────────────────────────┘
            │ (prompt [c_H ∘ p])
            ▼
[ Stage 4: Safety Verification ] ───► Decision ŷ ∈ { allow, block, defer }
            ├── y = allow ──► [ Stage 5: Environment Ledger Execution ]
            ├── y = block ──► [ Execution Blocked & Audit Alert ]
            └── y = defer ──► [ Bounded Read-Only Investigator Loop (up to 2 lookups) ]
```

### Selection Mechanisms Evaluated:
* **H0 (No History):** Empties historical context ($c_H = \emptyset$); tests decisions solely on the isolated tool proposal payload.
* **H1 (Recent History):** Greedily packs chronological events backwards from $e_t$ until budget $B=256$ tokens is exhausted.
* **H2 (BM25 Sparse):** Indexes trajectory events as text passages and ranks against proposal fields using BM25Okapi scoring.
* **H3 (Learned Reranking):** Retrieves the top-12 candidates via BM25, then reranks using a fine-tuned cross-encoder selector (`all-MiniLM-L6-v2` with LoRA).
* **H4 (Exhaustive Learned):** Directly scores all events in history $\mathcal{H}$ with the cross-encoder selector to select top-$K$ candidates.

---

## Repository Structure

```
.
├── Agent_Smith_Kaggle/              # Core Kaggle execution package and modular codebase
│   ├── Agent_Smith_Dataset_v1.zip   # Packaged synthetic dataset release v1.0
│   ├── Agent_Smith_Kaggle.ipynb     # Kaggle GPU execution notebook
│   ├── dataset.py                   # Procedural generation schemas, parsers, and loaders
│   ├── experiment.py                # Evaluation harness, metric collectors, and digest tools
│   ├── kaggle_run.py                # Command-line driver for multi-seed train and evaluate
│   ├── live.py                      # Multi-agent Qwen runner and fixture execution loop
│   ├── neural.py                    # PyTorch/LoRA MiniLM model architectures and trainers
│   ├── sandbox.py                   # Isolated mock environment, tool probes, and ledger
│   ├── smoke.py                     # GPU smoke test and environment validation script
│   ├── test_dataset.py              # Unit tests for policy verification and schemas
│   └── requirements.txt             # Python dependencies
│
├── agent_smith_hybrid.py            # Post-hoc hybrid replay and deterministic policy checker
├── Agent_Smith_Kaggle.ipynb         # Standalone root copy of the Kaggle notebook
├── Agent_Smith_Colab.ipynb          # Original Google Colab pilot replication notebook
├── Agent_Smith_Final_Proposal.pdf   # Original project proposal document
└── README.md                        # Project documentation
```

---

## Getting Started

### 1. Installation

Clone the repository and install the dependencies in a Python 3.10+ environment with CUDA PyTorch:

```bash
git clone git@github.com:shobhan-iiit/Agent_smith.git
cd Agent_smith/Agent_Smith_Kaggle
pip install -r requirements.txt
```

### 2. Run the GPU Smoke Test

Verify CUDA availability, pretrained base model loading, and tokenizer initialization:

```bash
python smoke.py
```

### 3. Run Training and Evaluation (Multi-Seed Kaggle Pipeline)

Execute the full 3-epoch, multi-seed training across family-disjoint folds:

```bash
export PYTHONHASHSEED=0

# Train cross-fitted selectors and monitors
python -u kaggle_run.py train --out ./agent_smith_run

# Evaluate on held-out test suite across all variants (H0–H4)
python -u kaggle_run.py evaluate --out ./agent_smith_run
```

### 4. Run the Hybrid Policy Verification Replay

Run the deterministic policy checker over the selected historical events to reproduce the 360/360 held-out accuracy and fixed-proposal investigator trials:

```bash
cd ..
python agent_smith_hybrid.py
```

---

## Synthetic Dataset Specifications

The benchmark evaluates six enterprise policy domains:
1. **Financial Transfers:** Spending limits, dual-authorization requirements, and beneficiary account registries.
2. **Confidential Data Exfiltration:** Scope-restricted datasets, classification levels, and destination IP filters.
3. **Infrastructure Deployment:** Approved container registries, resource quotas, and production lockouts.
4. **System Privilege Escalation:** Multi-factor authorization tokens, role-based access control, and time-bounded grants.
5. **Third-Party Package Publishing:** Registry credentials, version increment verification, and release freezes.
6. **Customer Record Modifications:** PII masking, opt-out flags, and GDPR deletion directives.

The 24/6/6 family split yields:
* **Train:** 1,440 rendered cases across 24 families (3,840 selector contrastive pairs; 7,200 out-of-fold monitor contexts).
* **Validation:** 360 rendered cases across 6 held-out families.
* **Held-Out Test:** 360 rendered cases across 6 held-out families (144 allow, 144 block, 72 defer; rendered at lengths 8, 16, and 32).

---

## Citation & References

```bibtex
@article{liu2024lost,
  title={Lost in the Middle: How Language Models Use Long Contexts},
  author={Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={Transactions of the Association for Computational Linguistics},
  volume={12},
  pages={157--173},
  year={2024}
}

@inproceedings{debenedetti2024agentdojo,
  title={{AgentDojo}: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for {LLM} Agents},
  author={Debenedetti, Edoardo and Zhang, Jie and Balunovi{\'c}, Mislav and Vazquez, Luca and Tram{\`e}r, Florian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
