from __future__ import annotations

"""
SQLM-style self-play trainer.

Usage:
    python local_rl_sqlm.py \
        --context_bank_path context_bank_mixed.jsonl \
        --output_dir ./runs/sqlm_shared \
        --policy_model_path Qwen/Qwen3-4B \
        --device cuda --use_lora --gradient_checkpointing \
        --train_steps 500 --batch_size 2 \
        --contexts_per_episode 5 --answerer_trials 7 \
        --majority_threshold 0.6
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_FINAL_LETTER_RE = re.compile(r"<final>\s*([ABCD])\s*</final>", flags=re.I)
_LETTER_ONLY_RE = re.compile(r"\b([ABCD])\b", flags=re.I)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

PROPOSER_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "object",
            "properties": {
                "A": {"type": "string"},
                "B": {"type": "string"},
                "C": {"type": "string"},
                "D": {"type": "string"},
            },
            "required": ["A", "B", "C", "D"],
            "additionalProperties": False,
        },
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "evidence_sentences": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["question", "options", "answer", "evidence_sentences"],
    "additionalProperties": False
}


def _dtype_from_string(s: str) -> torch.dtype:
    s = (s or "bfloat16").lower()
    if s == "float16":
        return torch.float16
    if s == "float32":
        return torch.float32
    return torch.bfloat16


def _get_rank() -> int:
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "PMI_RANK"):
        v = os.environ.get(k)
        if v is not None and str(v).isdigit():
            return int(v)
    return 0


def _clip(s: Any, max_chars: int = 4000) -> str:
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars] + "…[truncated]"


def _safe_json_load(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = _JSON_RE.search(s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _extract_final_letter(text: str) -> Optional[str]:
    m = _FINAL_LETTER_RE.search(text or "")
    if m:
        return m.group(1).upper()
    m2 = _LETTER_ONLY_RE.search((text or "").strip())
    return m2.group(1).upper() if m2 else None


def _iter_batches(xs: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def _log_paths(out_dir: str, prefix: str = "sqlm") -> Tuple[Path, Path, Path]:
    rank = _get_rank()
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"{prefix}_episodes_rank{rank}.jsonl",
        p / f"{prefix}_accepted_rank{rank}.jsonl",
        p / f"{prefix}_rejected_rank{rank}.jsonl",
    )


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class MCQProposal:
    question: str
    options: Dict[str, str]
    correct_letter: str
    evidence_sentences: List[str] = field(default_factory=list)
    prompt: str = ""
    raw_text: str = ""
    logprob: Optional[torch.Tensor] = None


@dataclass
class AnswerAttempt:
    letter: Optional[str]
    raw_text: str
    prompt: str
    logprob: Optional[torch.Tensor]


@dataclass
class EpisodeResult:
    contexts: List[str]
    context_ids: List[int]
    proposal: Optional[MCQProposal]
    attempts: List[AnswerAttempt]
    majority_letter: Optional[str]
    majority_count: int
    valid: bool
    proposer_reward: float
    solver_rewards: List[float]
    accepted: bool
    reject_reason: Optional[str] = None


# -----------------------------------------------------------------------------
# Context bank
# -----------------------------------------------------------------------------

_BANK_TEXTS: Optional[List[str]] = None
_BANK_IDS: Optional[List[int]] = None
_BANK_PATH_LOADED: Optional[str] = None


def load_context_bank(bank_path: str) -> Tuple[List[str], List[int]]:
    global _BANK_TEXTS, _BANK_IDS, _BANK_PATH_LOADED
    if _BANK_TEXTS is not None and _BANK_PATH_LOADED == bank_path:
        return _BANK_TEXTS, _BANK_IDS  # type: ignore

    p = Path(bank_path)
    if not p.exists():
        raise FileNotFoundError(f"Context bank not found: {bank_path}")

    texts: List[str] = []
    ids: List[int] = []

    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text") or obj.get("passage") or "").strip()
                if text:
                    texts.append(text)
                    ids.append(int(obj.get("id", i)))
    else:
        raw = p.read_text(encoding="utf-8")
        chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
        for i, c in enumerate(chunks):
            texts.append(c)
            ids.append(i)

    if not texts:
        raise ValueError(f"No contexts loaded from {bank_path}")

    _BANK_TEXTS, _BANK_IDS, _BANK_PATH_LOADED = texts, ids, bank_path
    return texts, ids


def sample_contexts_from_bank(
    bank_texts: List[str],
    bank_ids: List[int],
    n_ctx: int,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    if n_ctx >= len(bank_texts):
        idxs = list(range(len(bank_texts)))
    else:
        idxs = rng.sample(range(len(bank_texts)), n_ctx)
    return [bank_texts[i] for i in idxs], [bank_ids[i] for i in idxs]


# -----------------------------------------------------------------------------
# Prompt builders and parsing
# -----------------------------------------------------------------------------


def format_contexts(contexts: List[str], max_chars_each: int = 1800) -> str:
    out = []
    for i, c in enumerate(contexts):
        c2 = (c or "").strip()
        if len(c2) > max_chars_each:
            c2 = c2[:max_chars_each] + "…"
        out.append(f"[Context {i+1}]\n{c2}")
    return "\n\n".join(out)


def proposer_prompt(contexts: List[str]) -> str:
    ctx = format_contexts(contexts)
    return f"""You are the Proposer.
Using the information in the provided contexts, write EXACTLY ONE grounded 4-option multiple-choice question.
Return STRICT JSON only:
{{
  "question": "...",
  "options": {{"A":"...","B":"...","C":"...","D":"..."}},
  "answer": "A"|"B"|"C"|"D",
  "evidence_sentences": ["supporting sentence 1", "supporting sentence 2"]
}}
Constraints:
- The correct option must be supported by the contexts.
- Avoid pure copy/quote reading-comprehension when possible.
- Exactly one correct option.
- Provide multiple supporting sentences/evidence.
- Vary which letter (A/B/C/D) is the correct answer.
- Return JSON only.

CONTEXTS:
{ctx}
"""


def answerer_prompt(contexts: List[str], proposal: MCQProposal) -> str:
    ctx = format_contexts(contexts)
    o = proposal.options
    return f"""You are the Answerer.
Use the contexts to answer the multiple-choice question by choosing exactly one letter A/B/C/D.

Question:
{proposal.question}

Options:
A) {o['A']}
B) {o['B']}
C) {o['C']}
D) {o['D']}

CONTEXTS:
{ctx}

Output format:
<final>X</final>
"""


def parse_mcq_proposal(
    raw_text: str,
    prompt: str = "",
    logprob: Optional[torch.Tensor] = None,
) -> Optional[MCQProposal]:
    obj = _safe_json_load(raw_text)
    if not isinstance(obj, dict):
        return None

    q = str(obj.get("question") or "").strip()
    opts = obj.get("options")
    ans = str(obj.get("answer") or "").strip().upper()
    ev = obj.get("evidence_sentences") or []

    if not q or ans not in {"A", "B", "C", "D"}:
        return None

    if isinstance(opts, list) and len(opts) == 4:
        opts = {k: str(v).strip() for k, v in zip(["A", "B", "C", "D"], opts)}

    if not isinstance(opts, dict):
        return None

    norm_opts = {}
    for k in ["A", "B", "C", "D"]:
        v = str(opts.get(k) or "").strip()
        if not v:
            return None
        norm_opts[k] = v

    if not isinstance(ev, list):
        ev = []
    ev = [str(x).strip() for x in ev if str(x).strip()]

    return MCQProposal(
        question=q,
        options=norm_opts,
        correct_letter=ans,
        evidence_sentences=ev,
        prompt=prompt,
        raw_text=raw_text,
        logprob=logprob,
    )


def parse_answer_attempt(
    raw_text: str,
    prompt: str = "",
    logprob: Optional[torch.Tensor] = None,
) -> Optional[AnswerAttempt]:
    letter = _extract_final_letter(raw_text)
    if letter is None:
        return None
    return AnswerAttempt(letter=letter, raw_text=raw_text, prompt=prompt, logprob=logprob)


# -----------------------------------------------------------------------------
# Majority vote (SQLM core)
# -----------------------------------------------------------------------------


def majority_vote(letters: List[Optional[str]]) -> Tuple[Optional[str], int]:
    """Compute majority vote among valid letters.

    Returns:
        (majority_letter, majority_count)
        If no valid votes, returns (None, 0).
    """
    valid = [x for x in letters if x in {"A", "B", "C", "D"}]
    if not valid:
        return None, 0
    counts = Counter(valid)
    majority_letter, majority_count = counts.most_common(1)[0]
    return majority_letter, majority_count


# -----------------------------------------------------------------------------
# Local policy model wrapper (with all working fixes)
# -----------------------------------------------------------------------------

class LocalPolicyModel:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Collect all stop token IDs for chat models.
        stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        for name in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                stop_ids.add(tid)
        self.stop_token_ids = sorted(stop_ids) if stop_ids else [self.tokenizer.eos_token_id]

        model_dtype = _dtype_from_string(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=model_dtype,
        )
        self.model.to(device)
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as e:
                raise ImportError("Install peft to use --use_lora") from e
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.model = get_peft_model(self.model, peft_cfg)
            self.model.print_trainable_parameters()

        self.model.train()
        self._template_logged = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Encode a prompt using the model's chat template with thinking disabled."""
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt},
        ]
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, **template_kwargs, enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **template_kwargs)


        text = re.sub(r"<think>.*?</think>\s*$", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>\s*$", "", text)

        if not self._template_logged:
            self._template_logged = True
            print(f"[encode_prompt] template tail: ...{repr(text[-120:])}")
            print(f"[encode_prompt] stop_token_ids: {self.stop_token_ids}")

        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[str, torch.Tensor, Dict[str, torch.Tensor]]:
        enc = self.encode_prompt(prompt)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_token_ids,
            )
        if was_training:
            self.model.train()
        full_ids = out[0]
        prompt_len = enc["input_ids"].shape[1]
        gen_ids = full_ids[prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text, gen_ids, enc

    def sequence_logprob(self, prompt_inputs: Dict[str, torch.Tensor], gen_ids: torch.Tensor) -> torch.Tensor:
        """Compute sum-of-log-probs for generated tokens."""
        prompt_ids = prompt_inputs["input_ids"][0]
        gen_ids = gen_ids.to(prompt_ids.device)
        gen_len = gen_ids.shape[0]

        if gen_len == 0:
            return torch.tensor(0.0, device=prompt_ids.device, requires_grad=True)

        if self.gradient_checkpointing:
            full_ids = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0)
            outputs = self.model(input_ids=full_ids, use_cache=False)
            prompt_len = prompt_ids.shape[0]
            logits = outputs.logits[:, prompt_len - 1: prompt_len - 1 + gen_len, :]
            targets = gen_ids.unsqueeze(0)
        else:
            if prompt_ids.shape[0] > 1:
                with torch.no_grad():
                    prefix_out = self.model(
                        input_ids=prompt_ids[:-1].unsqueeze(0),
                        use_cache=True,
                    )
                    past_kv = prefix_out.past_key_values
            else:
                past_kv = None
            tail_ids = torch.cat([prompt_ids[-1:], gen_ids], dim=0).unsqueeze(0)
            tail_out = self.model(
                input_ids=tail_ids,
                past_key_values=past_kv,
                use_cache=False,
            )
            logits = tail_out.logits[:, :gen_len, :]
            targets = gen_ids.unsqueeze(0)

        token_logps = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        return token_logps.sum()

    def sample_with_logprob(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        text, gen_ids, enc = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        logprob = self.sequence_logprob(enc, gen_ids)
        return text, logprob

    def sample_json_with_logprob(
        self,
        prompt: str,
        schema: dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        return self.sample_with_logprob(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


# -----------------------------------------------------------------------------
# Training loop (SQLM: majority-vote validation)
# -----------------------------------------------------------------------------


def moving_average(old: float, new: float, momentum: float) -> float:
    if math.isnan(old):
        return new
    return momentum * old + (1.0 - momentum) * new


def train(args) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    bank_texts, bank_ids = load_context_bank(args.context_bank_path)

    policy_model_path = (
        args.policy_model_path
        or args.proposer_model_path
    )
    if policy_model_path is None:
        raise ValueError("Provide --policy_model_path.")

    policy = LocalPolicyModel(
        model_name_or_path=policy_model_path,
        device=args.device,
        dtype=args.dtype,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    policy_opt = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=args.policy_lr,
    )

    episodes_path, accepted_path, rejected_path = _log_paths(
        args.output_dir, prefix=args.log_prefix
    )

    prop_baseline = float("nan")
    ans_baseline = float("nan")

    policy.model.train()

    prop_denom = float(args.batch_size)
    ans_denom = float(args.batch_size * args.answerer_trials)

    for step in range(1, args.train_steps + 1):
        policy_opt.zero_grad(set_to_none=True)

        step_results: List[EpisodeResult] = []
        prop_loss_sum = 0.0
        ans_loss_sum = 0.0
        had_any_loss = False

        for b in range(args.batch_size):
            contexts, context_ids = sample_contexts_from_bank(
                bank_texts, bank_ids, args.contexts_per_episode, rng
            )

            # -------------------------
            # Proposer step
            # -------------------------
            p_prompt = proposer_prompt(contexts)
            p_text, p_logprob = policy.sample_json_with_logprob(
                prompt=p_prompt,
                schema=PROPOSER_SCHEMA,
                max_new_tokens=args.proposer_max_new_tokens,
                temperature=args.proposer_temperature,
                top_p=args.proposer_top_p,
                do_sample=True,
            )

            proposal = parse_mcq_proposal(p_text, logprob=p_logprob)
            if proposal is None:
                reject_payload = {
                    "ts": time.time(),
                    "step": step,
                    "context_ids": context_ids,
                    "contexts": [_clip(c, 1200) for c in contexts],
                    "reject_reason": "proposal_parse_failed",
                    "proposal_raw_text": _clip(p_text, 4000),
                }
                _append_jsonl(rejected_path, reject_payload)
                continue

            # -------------------------
            # Solver trials
            # -------------------------
            attempts: List[AnswerAttempt] = []
            solver_letters: List[Optional[str]] = []

            for _ in range(args.answerer_trials):
                a_prompt = answerer_prompt(contexts, proposal)
                a_text, a_logprob = policy.sample_with_logprob(
                    prompt=a_prompt,
                    max_new_tokens=args.answerer_max_new_tokens,
                    temperature=args.answerer_temperature,
                    top_p=args.answerer_top_p,
                    do_sample=True,
                )
                attempt = parse_answer_attempt(a_text, prompt=a_prompt, logprob=a_logprob)
                if attempt is None:
                    attempt = AnswerAttempt(letter=None, raw_text=a_text, prompt=a_prompt, logprob=a_logprob)
                attempts.append(attempt)
                solver_letters.append(attempt.letter)

            # -------------------------
            # SQLM majority-vote validation
            # -------------------------
            maj_letter, maj_count = majority_vote(solver_letters)
            N = args.answerer_trials
            n_valid_votes = sum(1 for x in solver_letters if x in {"A", "B", "C", "D"})

            has_majority = (
                maj_letter is not None
                and n_valid_votes >= 2
                and (maj_count / max(n_valid_votes, 1)) >= args.majority_threshold
            )
            not_trivial = 0 < maj_count < N
            valid = has_majority and not_trivial

            # -------------------------
            # SQLM rewards
            # -------------------------
            solver_rewards: List[float] = []
            for attempt in attempts:
                if valid and attempt.letter == maj_letter:
                    solver_rewards.append(1.0)
                else:
                    solver_rewards.append(0.0)

            proposer_reward = 1.0 if valid else 0.0

            # -------------------------
            # Baselines
            # -------------------------
            if valid:
                prop_baseline = moving_average(
                    prop_baseline, proposer_reward, args.baseline_momentum
                )
                ans_trial_mean = sum(solver_rewards) / float(max(len(solver_rewards), 1))
                ans_baseline = moving_average(
                    ans_baseline, ans_trial_mean, args.baseline_momentum
                )

            prop_adv = proposer_reward - (
                0.0 if math.isnan(prop_baseline) else prop_baseline
            )
            if proposal.logprob is not None:
                prop_loss_i = (
                    -proposal.logprob
                    * torch.tensor(
                        prop_adv / prop_denom,
                        device=proposal.logprob.device,
                        dtype=proposal.logprob.dtype,
                    )
                )
                prop_loss_i.backward()
                prop_loss_sum += float(prop_loss_i.detach().item())
                had_any_loss = True

            if valid:
                for attempt, r_sol in zip(attempts, solver_rewards):
                    if attempt.logprob is None:
                        continue
                    ans_adv = r_sol - (
                        0.0 if math.isnan(ans_baseline) else ans_baseline
                    )
                    ans_loss_i = (
                        -attempt.logprob
                        * torch.tensor(
                            args.answerer_loss_scale * ans_adv / ans_denom,
                            device=attempt.logprob.device,
                            dtype=attempt.logprob.dtype,
                        )
                    )
                    ans_loss_i.backward()
                    ans_loss_sum += float(ans_loss_i.detach().item())
                    had_any_loss = True

            # -------------------------
            # Logging
            # -------------------------
            ep = EpisodeResult(
                contexts=contexts,
                context_ids=context_ids,
                proposal=proposal,
                attempts=attempts,
                majority_letter=maj_letter,
                majority_count=maj_count,
                valid=valid,
                proposer_reward=proposer_reward,
                solver_rewards=solver_rewards,
                accepted=bool(valid),
                reject_reason=None if valid else (
                    "no_majority" if not has_majority else "unanimous"
                ),
            )
            step_results.append(ep)

            log_payload = {
                "ts": time.time(),
                "step": step,
                "context_ids": context_ids,
                "contexts": [_clip(c, 1200) for c in contexts],
                "proposal": {
                    "question": proposal.question,
                    "options": proposal.options,
                    "correct_letter": proposal.correct_letter,
                    "evidence_sentences": proposal.evidence_sentences,
                    "raw_text": _clip(proposal.raw_text, 4000),
                },
                "majority_letter": maj_letter,
                "majority_count": maj_count,
                "n_trials": N,
                "solver_letters": solver_letters,
                "attempts": [
                    {"letter": a.letter, "raw_text": _clip(a.raw_text, 1000)}
                    for a in attempts
                ],
                "valid": valid,
                "proposer_reward": proposer_reward,
                "solver_rewards": solver_rewards,
            }
            _append_jsonl(accepted_path if valid else rejected_path, log_payload)

        if not had_any_loss:
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), args.max_grad_norm)

        policy_opt.step()

        mean_valid = sum(1.0 for ep in step_results if ep.valid) / float(max(len(step_results), 1))
        mean_prop_r = sum(ep.proposer_reward for ep in step_results) / float(max(len(step_results), 1))
        mean_maj = sum(
            (ep.majority_count / float(max(args.answerer_trials, 1))) for ep in step_results
        ) / float(max(len(step_results), 1))
        summary = {
            "ts": time.time(),
            "step": step,
            "prop_loss": prop_loss_sum,
            "ans_loss": ans_loss_sum,
            "mean_valid": mean_valid,
            "mean_prop_reward": mean_prop_r,
            "mean_majority_frac": mean_maj,
            "prop_baseline": None if math.isnan(prop_baseline) else prop_baseline,
            "ans_baseline": None if math.isnan(ans_baseline) else ans_baseline,
        }
        _append_jsonl(episodes_path, summary)
        if _get_rank() == 0:
            print(json.dumps(summary, ensure_ascii=False))

        if args.save_every > 0 and step % args.save_every == 0:
            policy.save(str(Path(args.output_dir) / f"policy_step{step}"))

    policy.save(str(Path(args.output_dir) / "policy_final"))



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SQLM-style self-play trainer (majority-vote validation)")
    p.add_argument("--context_bank_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./runs/sqlm_selfplay")
    p.add_argument("--log_prefix", type=str, default="sqlm")

    p.add_argument("--policy_model_path", type=str, default=None)
    p.add_argument("--proposer_model_path", type=str, default=None)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--train_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--contexts_per_episode", type=int, default=5)
    p.add_argument("--answerer_trials", type=int, default=7,
                    help="Number of solver rollouts per question (odd number recommended)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline_momentum", type=float, default=0.9)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=20)

    p.add_argument("--policy_lr", type=float, default=1e-5)
    p.add_argument("--answerer_loss_scale", type=float, default=1.0)

    p.add_argument("--proposer_max_new_tokens", type=int, default=768)
    p.add_argument("--proposer_temperature", type=float, default=0.8)
    p.add_argument("--proposer_top_p", type=float, default=0.95)

    p.add_argument("--answerer_max_new_tokens", type=int, default=32)
    p.add_argument("--answerer_temperature", type=float, default=0.7)
    p.add_argument("--answerer_top_p", type=float, default=0.95)

    p.add_argument("--majority_threshold", type=float, default=0.6,
                    help="Minimum fraction of votes for the majority letter to count as valid (e.g. 0.6 = 60%%)")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()