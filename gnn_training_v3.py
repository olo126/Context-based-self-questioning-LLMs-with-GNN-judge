"""
Usage:
    python gnn_training_v3.py \
        --output_dir outputs/hotpot_gnn_v3 \
        --max_train_examples 20000 \
        --max_val_examples 4000 \
        --epochs 20 \
        --batch_size 32 \
        --neg_ratio 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_yes_no_answer(ans: str) -> bool:
    return normalize_text(ans) in {"yes", "no"}


_NP_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)


def extract_entities(text: str) -> List[str]:
    """Extract capitalized noun phrases as candidate entities."""
    return list(set(_NP_RE.findall(text)))


# -----------------------------------------------------------------------------
# HotpotQA -> binary support examples
# -----------------------------------------------------------------------------

@dataclass
class SupportExample:
    qid: str
    question: str
    candidate_answer: str
    evidence_sentences: List[str]
    label: int
    source_answer: str
    meta: Dict[str, str]


class HotpotSupportBuilder:
    """
    Builds support-verification examples from HotpotQA.
    """

    def __init__(
        self,
        max_extra_distractor_sents: int = 2,
        min_support_sentences: int = 1,
        answer_pool: Optional[List[str]] = None,
        seed: int = 42,
        mcq_augment: bool = True,
    ) -> None:
        self.max_extra_distractor_sents = max_extra_distractor_sents
        self.min_support_sentences = min_support_sentences
        self.answer_pool = answer_pool or []
        self.rng = random.Random(seed)
        self.mcq_augment = mcq_augment

    def _context_map(self, ex: dict) -> Dict[str, List[str]]:
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        return {title: para_sents for title, para_sents in zip(titles, sentences)}

    def _supporting_sentences(self, ex: dict) -> List[str]:
        ctx = self._context_map(ex)
        sents: List[str] = []
        seen = set()
        for title, sent_id in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
            para = ctx.get(title, [])
            if 0 <= sent_id < len(para):
                sent = para[sent_id].strip()
                if sent and sent not in seen:
                    sents.append(sent)
                    seen.add(sent)
        return sents

    def _all_context_sentences(self, ex: dict) -> List[str]:
        all_sents = []
        for para_sents in ex["context"]["sentences"]:
            for sent in para_sents:
                sent = sent.strip()
                if sent:
                    all_sents.append(sent)
        return all_sents

    def _nonsupport_sentences(self, ex: dict, support_sents: List[str]) -> List[str]:
        support_set = set(support_sents)
        return [s for s in self._all_context_sentences(ex) if s not in support_set]

    def _context_derived_wrong_answers(self, ex: dict, gold_answer: str, max_candidates: int = 5) -> List[str]:
        """Extract plausible wrong answers from the context paragraphs."""
        gold_norm = normalize_text(gold_answer)
        all_text = " ".join(self._all_context_sentences(ex))
        entities = extract_entities(all_text)

        # Filter out the gold answer and very short entities
        candidates = [
            e for e in entities
            if normalize_text(e) != gold_norm and len(e) > 2
        ]
        self.rng.shuffle(candidates)
        return candidates[:max_candidates]

    def _sample_wrong_answer_hard(self, ex: dict, gold_answer: str) -> str:
        """Try context-derived wrong answer first, fall back to random pool."""
        gold_norm = normalize_text(gold_answer)
        if is_yes_no_answer(gold_answer):
            return "no" if gold_norm == "yes" else "yes"

        # Try context-derived entities first
        candidates = self._context_derived_wrong_answers(ex, gold_answer, max_candidates=10)
        if candidates:
            return self.rng.choice(candidates)

        # Fallback to global pool
        pool = [a for a in self.answer_pool if normalize_text(a) != gold_norm]
        if not pool:
            return gold_answer + " not"
        return self.rng.choice(pool)

    def _to_sentence_form(self, question: str, answer: str) -> str:
        """Convert a short extractive answer into a sentence-form MCQ option."""
        q = question.rstrip("?").strip()
        return f"{answer}"

    def build(self, ex: dict) -> List[SupportExample]:
        qid = ex.get("id", ex.get("_id", ""))
        question = ex["question"].strip()
        answer = ex["answer"].strip()
        support_sents = self._supporting_sentences(ex)

        if len(support_sents) < self.min_support_sentences:
            return []

        out: List[SupportExample] = []

        # ---- Positive: gold answer + gold evidence ----
        out.append(
            SupportExample(
                qid=qid, question=question, candidate_answer=answer,
                evidence_sentences=support_sents, label=1,
                source_answer=answer,
                meta={"kind": "positive_gold_answer_gold_evidence"},
            )
        )

        # ---- Negative 1: HARD wrong answer (context-derived) + gold evidence ----
        hard_wrong = self._sample_wrong_answer_hard(ex, answer)
        out.append(
            SupportExample(
                qid=qid, question=question, candidate_answer=hard_wrong,
                evidence_sentences=support_sents, label=0,
                source_answer=answer,
                meta={"kind": "negative_hard_wrong_answer_same_evidence"},
            )
        )

        # ---- Negative 2: second hard wrong answer if available ----
        candidates = self._context_derived_wrong_answers(ex, answer, max_candidates=5)
        second_wrong = None
        for c in candidates:
            if normalize_text(c) != normalize_text(hard_wrong):
                second_wrong = c
                break
        if second_wrong:
            out.append(
                SupportExample(
                    qid=qid, question=question, candidate_answer=second_wrong,
                    evidence_sentences=support_sents, label=0,
                    source_answer=answer,
                    meta={"kind": "negative_hard_wrong_answer2_same_evidence"},
                )
            )

        # ---- Negative 3: gold answer + distractor evidence ----
        nonsupport = self._nonsupport_sentences(ex, support_sents)
        if nonsupport:
            k = min(len(support_sents) + self.max_extra_distractor_sents, len(nonsupport))
            distractor_sents = self.rng.sample(nonsupport, k=k)
            out.append(
                SupportExample(
                    qid=qid, question=question, candidate_answer=answer,
                    evidence_sentences=distractor_sents, label=0,
                    source_answer=answer,
                    meta={"kind": "negative_gold_answer_distractor_evidence"},
                )
            )

        # ---- MCQ augmentation: sentence-form versions ----
        if self.mcq_augment and not is_yes_no_answer(answer):
            sent_answer = self._to_sentence_form(question, answer)
            if sent_answer != answer:
                out.append(
                    SupportExample(
                        qid=qid, question=question, candidate_answer=sent_answer,
                        evidence_sentences=support_sents, label=1,
                        source_answer=answer,
                        meta={"kind": "positive_sentence_form"},
                    )
                )
            sent_wrong = self._to_sentence_form(question, hard_wrong)
            if sent_wrong != hard_wrong:
                out.append(
                    SupportExample(
                        qid=qid, question=question, candidate_answer=sent_wrong,
                        evidence_sentences=support_sents, label=0,
                        source_answer=answer,
                        meta={"kind": "negative_sentence_form_wrong"},
                    )
                )

        return out


# -----------------------------------------------------------------------------
# Graph collation
# -----------------------------------------------------------------------------

class TextGraphCollator:
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        max_evidence_sentences: int = 6,
    ) -> None:
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.max_evidence_sentences = max_evidence_sentences

    def _make_graph(self, ex: SupportExample) -> Data:
        evidence_sents = ex.evidence_sentences[: self.max_evidence_sentences]
        claim = f"Question: {ex.question}\nCandidate answer: {ex.candidate_answer}"
        texts = [claim] + evidence_sents
        if len(texts) == 1:
            texts.append(claim)

        emb = self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        x = torch.tensor(emb, dtype=torch.float)
        n = x.size(0)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y = torch.tensor([float(ex.label)], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

    def __call__(self, batch: Sequence[SupportExample]) -> Batch:
        graphs = [self._make_graph(ex) for ex in batch]
        return Batch.from_data_list(graphs)


# -----------------------------------------------------------------------------
# GNN model
# -----------------------------------------------------------------------------

class SupportGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        pooled = global_mean_pool(x, batch.batch)
        logits = self.cls(pooled).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

class SupportTorchDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SupportExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SupportExample:
        return self.examples[idx]


def build_support_dataset(
    split: Dataset,
    answer_pool: List[str],
    max_source_examples: Optional[int],
    max_extra_distractor_sents: int,
    seed: int,
    mcq_augment: bool = True,
) -> List[SupportExample]:
    builder = HotpotSupportBuilder(
        max_extra_distractor_sents=max_extra_distractor_sents,
        answer_pool=answer_pool,
        seed=seed,
        mcq_augment=mcq_augment,
    )
    n = len(split) if max_source_examples is None else min(max_source_examples, len(split))
    all_examples: List[SupportExample] = []
    for i in tqdm(range(n), desc="building support examples"):
        ex = split[i]
        all_examples.extend(builder.build(ex))
    return all_examples


def balance_examples(
    examples: List[SupportExample],
    strategy: str = "undersample",
    ratio: float = 1.0,
    seed: int = 42,
) -> List[SupportExample]:
    """Balance positive/negative examples.

    Args:
        strategy: 'undersample' negatives or 'oversample' positives.
        ratio: target neg:pos ratio. 1.0 = equal, 1.5 = 1.5x negatives per positive.
        seed: random seed for reproducibility.
    """
    rng = random.Random(seed)
    positives = [ex for ex in examples if ex.label == 1]
    negatives = [ex for ex in examples if ex.label == 0]
    n_pos = len(positives)
    n_neg = len(negatives)
    target_neg = int(n_pos * ratio)

    if strategy == "undersample":
        if n_neg > target_neg:
            rng.shuffle(negatives)
            negatives = negatives[:target_neg]
    elif strategy == "oversample":
        if n_pos < n_neg:
            target_pos = int(n_neg / ratio)
            while len(positives) < target_pos:
                positives.append(rng.choice(positives))

    combined = positives + negatives
    rng.shuffle(combined)
    return combined


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Metrics:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels_i = labels.long()
    tp = int(((preds == 1) & (labels_i == 1)).sum().item())
    tn = int(((preds == 0) & (labels_i == 0)).sum().item())
    fp = int(((preds == 1) & (labels_i == 0)).sum().item())
    fn = int(((preds == 0) & (labels_i == 1)).sum().item())
    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return Metrics(loss=0.0, accuracy=acc, precision=precision, recall=recall, f1=f1)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Metrics:
    training = optimizer is not None
    model.train(training)

    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_graphs = 0
    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y.view(-1)
        loss = loss_fn(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = batch.num_graphs
        total_loss += float(loss.item()) * bs
        total_graphs += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if training and scheduler is not None:
        scheduler.step()

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_binary_metrics(logits_cat, labels_cat)
    metrics.loss = total_loss / max(total_graphs, 1)
    return metrics


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------

class HotpotVerifier:
    def __init__(self, checkpoint_path: str, encoder_name: str, device: str = "cpu") -> None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.device = device
        self.collator = TextGraphCollator(encoder_name=encoder_name, device=device)
        self.model = SupportGAT(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, question: str, candidate_answer: str, evidence_sentences: List[str]) -> float:
        ex = SupportExample(
            qid="inference", question=question,
            candidate_answer=candidate_answer,
            evidence_sentences=evidence_sentences,
            label=0, source_answer="",
            meta={"kind": "inference"},
        )
        batch = self.collator([ex]).to(self.device)
        logit = self.model(batch)
        return float(torch.sigmoid(logit).item())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_train_examples", type=int, default=20000)
    ap.add_argument("--max_val_examples", type=int, default=4000)
    ap.add_argument("--max_extra_distractor_sents", type=int, default=2)
    ap.add_argument("--max_evidence_sentences", type=int, default=6)
    ap.add_argument("--no_mcq_augment", action="store_true",
                    help="Disable MCQ sentence-form augmentation")
    ap.add_argument("--neg_ratio", type=float, default=1.0,
                    help="Target negative:positive ratio after undersampling (1.0 = balanced)")
    ap.add_argument("--no_balance", action="store_true",
                    help="Disable negative undersampling")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (0 to disable)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/6] Loading HotpotQA distractor split...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    train_split = ds["train"]
    val_split = ds["validation"]

    print("[2/6] Building answer pool...")
    answer_pool = []
    for i in range(min(len(train_split), args.max_train_examples)):
        a = train_split[i]["answer"].strip()
        if a:
            answer_pool.append(a)

    mcq_augment = not args.no_mcq_augment

    print("[3/6] Converting HotpotQA train split into support examples...")
    train_examples = build_support_dataset(
        split=train_split,
        answer_pool=answer_pool,
        max_source_examples=args.max_train_examples,
        max_extra_distractor_sents=args.max_extra_distractor_sents,
        seed=args.seed,
        mcq_augment=mcq_augment,
    )

    from collections import Counter
    type_counts = Counter(ex.meta["kind"] for ex in train_examples)
    n_pos = sum(1 for ex in train_examples if ex.label == 1)
    n_neg = sum(1 for ex in train_examples if ex.label == 0)
    print(f"Built {len(train_examples)} training examples ({n_pos} pos, {n_neg} neg, ratio={n_neg/max(n_pos,1):.1f}:1):")
    for k, v in type_counts.most_common():
        print(f"  {k}: {v}")


    if not args.no_balance:
        train_examples = balance_examples(
            train_examples,
            strategy="undersample",
            ratio=args.neg_ratio,
            seed=args.seed,
        )
        n_pos_b = sum(1 for ex in train_examples if ex.label == 1)
        n_neg_b = sum(1 for ex in train_examples if ex.label == 0)
        print(f"After balancing (ratio={args.neg_ratio}): {len(train_examples)} examples ({n_pos_b} pos, {n_neg_b} neg)")

    print("[4/6] Converting HotpotQA validation split into support examples...")
    val_examples = build_support_dataset(
        split=val_split,
        answer_pool=answer_pool,
        max_source_examples=args.max_val_examples,
        max_extra_distractor_sents=args.max_extra_distractor_sents,
        seed=args.seed + 1,
        mcq_augment=mcq_augment,
    )

    if not args.no_balance:
        val_examples = balance_examples(
            val_examples,
            strategy="undersample",
            ratio=args.neg_ratio,
            seed=args.seed + 1,
        )
    n_vp = sum(1 for ex in val_examples if ex.label == 1)
    n_vn = sum(1 for ex in val_examples if ex.label == 0)
    print(f"Built {len(val_examples)} validation examples ({n_vp} pos, {n_vn} neg).")


    with open(Path(args.output_dir) / "train_examples_preview.jsonl", "w", encoding="utf-8") as f:
        for ex in train_examples[:200]:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    train_dataset = SupportTorchDataset(train_examples)
    val_dataset = SupportTorchDataset(val_examples)

    collator = TextGraphCollator(
        encoder_name=args.encoder_name,
        device=args.device,
        max_evidence_sentences=args.max_evidence_sentences,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator,
    )

    print("[5/6] Initializing GNN...")
    sample_batch = next(iter(train_loader)).to(args.device)
    input_dim = sample_batch.x.size(-1)
    model = SupportGAT(input_dim=input_dim, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    print(f"[6/6] Training for up to {args.epochs} epochs (patience={args.patience})...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device, scheduler)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, args.device)

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(row)
        improved = val_metrics.f1 > best_val_f1
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:2d} | lr={lr_now:.2e} | "
            f"train_loss={train_metrics.loss:.4f} train_f1={train_metrics.f1:.3f} | "
            f"val_loss={val_metrics.loss:.4f} val_f1={val_metrics.f1:.3f} "
            f"val_acc={val_metrics.accuracy:.3f} val_prec={val_metrics.precision:.3f} "
            f"val_rec={val_metrics.recall:.3f}{marker}"
        )

        if improved:
            best_val_f1 = val_metrics.f1
            patience_counter = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "encoder_name": args.encoder_name,
                "best_val_f1": best_val_f1,
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, Path(args.output_dir) / "best_model.pt")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    with open(Path(args.output_dir) / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Saved best checkpoint to: {Path(args.output_dir) / 'best_model.pt'}")
    print(f"Saved metrics to: {Path(args.output_dir) / 'metrics_history.json'}")


if __name__ == "__main__":
    main()