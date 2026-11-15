# Agent Smith on Kaggle

1. Download the release ZIP and extract it. Import `Agent_Smith_Kaggle.ipynb` into Kaggle.
2. Upload the **original release ZIP** as a private Kaggle Dataset. Attach that dataset to the notebook. Keep the name `Agent_Smith_Kaggle.zip`.
3. Turn on **Internet** and select **GPU T4 x2** or another GPU accelerator. The code uses one GPU. The Internet connection is for dependencies and pretrained weights.
4. Run the setup and GPU smoke test, then training, evaluation, and export. The final artifact is `/kaggle/working/Agent_Smith_Kaggle_Results.zip`. Download it from the notebook output or save a notebook version with outputs.

The notebook defaults to two seeds, three epochs, batch 16, and all included ablations. If session time is tight, set `SEEDS = [17]` before the first training run and keep three epochs. If GPU memory runs out during fitting, start a new run directory with `BATCH = 8`. Smoke output verifies loading only. It contains no benchmark result.

## Run without the notebook

From this directory in an environment with CUDA PyTorch 2.6 or later and the usual scientific Python stack:

```bash
python -m pip install -r requirements.txt
export PYTHONHASHSEED=0
python smoke.py
python -u kaggle_run.py train --out /kaggle/working/agent_smith_run
python -u kaggle_run.py evaluate --out /kaggle/working/agent_smith_run
```

Keep the same arguments in both phases. Re-run training with the same output directory to skip completed model stages and reuse completed fold exports. Interrupted training restarts the current stage. To resume after a Kaggle session ends, first recover the run directory from saved outputs or a prior downloaded results archive. The freeze manifest binds the data, configuration, model checkpoints and source code.

## What runs

- MiniLM/LoRA rank-8 selector and three-class monitor, three family-held-out selector folds, and MiniLM tokenizer limits of 256 history and 512 total tokens.
- H0–H4, grouping and reranker policy-feature ablations, frozen-encoder selector and monitor baselines, an independent final test stage, per-case predictions, family breakdowns and paired accuracy intervals.
- Frozen Qwen2.5-1.5B-Instruct worker and investigator on eight held-out synthetic fixtures. Includes handoff A0–A3, policy retrieval P0–P2 with a trained reranker, reasoning Q0–Q2, and tool choice/retry T0–T2. The mock controller requires sufficient observed policy evidence for an allow, and writes only simulated effects.
- Model histories, save/reload differences, policy retrieval metrics, replay accuracy, unsafe allows, deferrals, evidence coverage, scoped tool outcomes and audit traces.

The worker is given a constrained, fixture-defined action. Handoff variants use local delivery. The current fixture counts differ from the proposal's planned counts. Q and P comparisons are episode-level and can affect which evidence is retrieved, so interpret them as end-to-end contrasts. Labels remain procedural and provisional pending independent team review. The package includes experiment code and data. Use the metrics produced by Kaggle for any neural performance claims.
