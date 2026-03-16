# Context-based-self-questioning-LLMs-with-GNN-judge
CSE 515 project

This project trains an LLM to generate and answer grounded multiple-choice questions from context passages using self-play reinforcement learning. Two training pipelines are provided: one using a GNN judge for validation, and one using SQLM-style majority-vote validation.

## Requirements

```bash
pip install torch transformers peft datasets sentence-transformers torch-geometric tqdm
```

## Project Structure

```
├── make_bank.py              # Build context bank from FineWeb + OpenWebMath
├── gnn_training_v3.py        # Train GNN evidence verifier on HotpotQA
├── sq_rl_v4.py               # Self-play training with GNN judge
├── sq_rl_sqlm.py             # Self-play training with majority-vote (SQLM)
├── eval_GSM8K.py             # Evaluate on GSM8K math benchmark
├── eval_mmlu_subj.py         # Evaluate on MMLU benchmark
└── README.md
```

---

## Step 1: Build the Context Bank

Creates a mixed context bank from FineWeb (general web text) and OpenWebMath (math content).

```bash
python make_bank.py \
    --out context_bank_mixed.jsonl \
    --fineweb-docs 2500 \
    --openwebmath-docs 2500 \
    --seed 0 \
    --max-chars 8000 \
    --require-min-chars 300
```

**Output:** `context_bank_mixed.jsonl` — 5000 context passages in JSONL format.

---

## Step 2: Train the GNN Judge (for GNN pipeline only)

Trains a GAT-based evidence verifier on HotpotQA. This step is only needed if using `sq_rl_v4.py`.

```bash
python gnn_training_v3.py \
    --output_dir outputs/hotpot_gnn_v3 \
    --max_train_examples 20000 \
    --max_val_examples 4000 \
    --epochs 20 \
    --batch_size 32 \
    --neg_ratio 1.0
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--neg_ratio` | 1.0 | Negative-to-positive ratio after undersampling |
| `--no_balance` | false | Disable class balancing |
| `--patience` | 5 | Early stopping patience based on val F1 |
| `--epochs` | 20 | Maximum training epochs |

**Output:** `outputs/hotpot_gnn_v3/best_model.pt` — best checkpoint by val F1.

---

## Step 3a: Self-Play Training with GNN Judge

The proposer generates an MCQ with evidence from context passages. The GNN judge verifies that the evidence supports the claimed answer. The answerer attempts the MCQ multiple times. The proposer is rewarded for medium-difficulty valid questions; the answerer is rewarded for correct answers.

```bash
python sq_rl_v4.py \
    --context_bank_path context_bank_mixed.jsonl \
    --output_dir ./runs/gnn_selfplay \
    --policy_model_path Qwen/Qwen3-4B \
    --judge_type gnn \
    --gnn_judge_ckpt outputs/hotpot_gnn_v3/best_model.pt \
    --device cuda \
    --judge_device cuda \
    --use_lora \
    --gradient_checkpointing \
    --train_steps 150 \
    --batch_size 2 \
    --contexts_per_episode 5 \
    --answerer_trials 5 \
    --policy_lr 1e-5 \
    --save_every 50
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy_model_path` | required | Base model (e.g. `Qwen/Qwen3-4B`) |
| `--judge_type` | `gnn` | Judge type: `gnn` or `llm` |
| `--gnn_judge_ckpt` | None | Path to trained GNN checkpoint |
| `--judge_tau` | 0.7 | Minimum GNN score threshold for validation |
| `--use_lora` | false | Enable LoRA fine-tuning |
| `--gradient_checkpointing` | false | Enable gradient checkpointing |
| `--answerer_trials` | 5 | Number of solver rollouts per question |
| `--proposer_max_new_tokens` | 512 | Max generation length for proposer |
| `--answerer_max_new_tokens` | 256 | Max generation length for answerer |
| `--answerer_loss_scale` | 1.0 | Weight of answerer loss relative to proposer |

**Outputs:**
- `runs/gnn_selfplay/policy_step{N}/` — LoRA checkpoints every N steps
- `runs/gnn_selfplay/policy_final/` — final checkpoint
- `runs/gnn_selfplay/selfplay_episodes_rank0.jsonl` — per-step training summaries
- `runs/gnn_selfplay/selfplay_accepted_rank0.jsonl` — accepted episode details
- `runs/gnn_selfplay/selfplay_rejected_rank0.jsonl` — rejected episode details

---

## Step 3b: Self-Play Training with Majority Vote (SQLM)

Based on [Self-Questioning Language Models (Chen et al., 2025)](https://arxiv.org/abs/2508.03682). No external judge — the majority vote among solver rollouts serves as the proxy gold label.

```bash
python sq_rl_sqlm.py \
    --context_bank_path context_bank_mixed.jsonl \
    --output_dir ./runs/sqlm_selfplay \
    --policy_model_path Qwen/Qwen3-4B \
    --device cuda \
    --use_lora \
    --gradient_checkpointing \
    --train_steps 150 \
    --batch_size 2 \
    --contexts_per_episode 5 \
    --answerer_trials 7 \
    --majority_threshold 0.6 \
    --policy_lr 1e-5 \
    --save_every 20
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--answerer_trials` | 7 | Number of solver rollouts (odd recommended) |
| `--majority_threshold` | 0.6 | Minimum fraction of votes for majority to be valid |

**Outputs:** Same structure as GNN pipeline with `sqlm_` prefix on log files.

---

## Step 4: Evaluation

### GSM8K (Math Reasoning)

```bash
# Evaluate trained LoRA checkpoint
python eval_GSM8K.py \
    --output_dir ./eval_results/gsm8k \
    --model_path ./runs/gnn_selfplay/policy_final \
    --base_model_path Qwen/Qwen3-4B \
    --batch_size 16

# Evaluate base model (baseline)
python eval_GSM8K.py \
    --output_dir ./eval_results/gsm8k_base \
    --model_path Qwen/Qwen3-4B \
    --batch_size 16
```

### MMLU (General Knowledge)

```bash
# Evaluate trained LoRA checkpoint
python eval_mmlu_subj.py \
    --output_dir ./eval_results/mmlu \
    --model_path ./runs/gnn_selfplay/policy_final \
    --base_model_path Qwen/Qwen3-4B \
    --batch_size 32

# Evaluate base model (baseline)
python eval_mmlu_subj.py \
    --output_dir ./eval_results/mmlu_base \
    --model_path Qwen/Qwen3-4B \
    --batch_size 32
```

**Key arguments for both eval scripts:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | required | Path to LoRA adapter directory or full model |
| `--base_model_path` | None | Base model for LoRA loading (e.g. `Qwen/Qwen3-4B`). Omit for full models. |
| `--batch_size` | 16/32 | Evaluation batch size |

**Outputs:**
- `eval_results/gsm8k/gsm8k_score_{model}.json` — accuracy summary
- `eval_results/gsm8k/gsm8k_responses_{model}.jsonl` — per-question responses
- `eval_results/mmlu/mmlu_scores_{model}.json` — overall + per-subject accuracy
- `eval_results/mmlu/responses_{model}.jsonl` — per-question responses

---

## Full Pipeline Example

```bash
# 1. Build context bank
python make_bank.py --out context_bank_mixed.jsonl

# 2. Train GNN judge
python gnn_training_v3.py --output_dir outputs/hotpot_gnn_v3

# 3. Run self-play training (GNN judge variant)
python sq_rl_v4.py \
    --context_bank_path context_bank_mixed.jsonl \
    --output_dir ./runs/gnn_selfplay \
    --policy_model_path Qwen/Qwen3-4B \
    --judge_type gnn \
    --gnn_judge_ckpt outputs/hotpot_gnn_v3/best_model.pt \
    --device cuda --judge_device cuda \
    --use_lora --gradient_checkpointing \
    --train_steps 150 --batch_size 2

# 4. Evaluate
python eval_GSM8K.py --output_dir ./eval_results/gsm8k \
    --model_path ./runs/gnn_selfplay/policy_final \
    --base_model_path Qwen/Qwen3-4B

python eval_mmlu_subj.py --output_dir ./eval_results/mmlu \
    --model_path ./runs/gnn_selfplay/policy_final \
    --base_model_path Qwen/Qwen3-4B
```

---
