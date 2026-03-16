import torch, argparse, os, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

def encode_prompt(tokenizer, prompt_text):
    """Apply chat template with thinking disabled for Qwen3."""
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": prompt_text},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    text = re.sub(r"<think>.*?</think>\s*$", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>\s*$", "", text)
    return text

def get_stop_token_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return sorted(stop_ids)

def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def make_prompt(question, choices):
    letters = ["A", "B", "C", "D"]
    options = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return (
        f"Question: {question}\n"
        f"{options}\n"
        "Choose the correct option and answer with only the letter A, B, C, or D. "
        "Report your answer with \"Final Answer: \"\n"
        "Answer: "
    )

def extract_letter(output: str):
    if not output:
        return None
    text = output.strip()

    match = re.search(r"(?i)final\s*answer\s*[:\-\*]*\s*\(?([ABCD])\)?", text)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?i)the\s*answer\s*is\s*\(?([ABCD])\)?", text)
    if match:
        return match.group(1).upper()

    match = re.findall(r"(?i)option\s*([ABCD])", text)
    if match:
        return match[-1].upper()

    return None

def load_model(args):
    """Load model, handling both full models and LoRA adapters."""
    model_path = args.model_path.rstrip("/")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.base_model_path:
        print(f"Loading base model: {args.base_model_path}")
        print(f"Loading LoRA adapter: {model_path}")
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged.")
    else:
        print(f"Loading full model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer

def main(args):
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")

    model, tokenizer = load_model(args)
    stop_ids = get_stop_token_ids(tokenizer)

    batch_size = args.batch_size
    os.makedirs(args.output_dir, exist_ok=True)
    model_name_safe = os.path.basename(args.model_path.rstrip("/")).replace('/', '_')
    record_path = os.path.join(args.output_dir, f"responses_{model_name_safe}.jsonl")

    overall_correct = 0
    overall_total = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    print(f"Starting evaluation (Batch size: {batch_size})...")

    with open(record_path, "w", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[start:start + batch_size]

            questions = batch["question"]
            choices_list = batch["choices"]
            answers = batch["answer"]
            subjects = batch["subject"]

            raw_prompts = [make_prompt(q, c) for q, c in zip(questions, choices_list)]
            templated_prompts = [encode_prompt(tokenizer, p) for p in raw_prompts]

            enc = tokenizer(
                templated_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids,
                )

            max_input_len = enc["input_ids"].shape[1]

            for i in range(len(questions)):
                gen_ids = gen[i, max_input_len:]
                out = tokenizer.decode(gen_ids, skip_special_tokens=True)
                out = strip_think(out)

                pred = extract_letter(out)
                gold_idx = answers[i]
                gold = "ABCD"[gold_idx]
                subject = subjects[i]

                is_correct = (pred == gold)

                overall_total += 1
                subject_stats[subject]["total"] += 1
                if is_correct:
                    overall_correct += 1
                    subject_stats[subject]["correct"] += 1

                record = {
                    "subject": subject,
                    "correct": is_correct,
                    "gold": gold,
                    "pred": pred,
                    "response": out,
                    "prompt": raw_prompts[i],
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    per_subject_results = {}
    for subj, stats in subject_stats.items():
        if stats["total"] > 0:
            per_subject_results[subj] = {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"]
            }

    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_correct}/{overall_total})")

    out_path = os.path.join(args.output_dir, f"mmlu_scores_{model_name_safe}.json")
    final_results = {
        "overall_accuracy": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "subject_scores": per_subject_results
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_path', required=True,
                        help="Path to checkpoint (LoRA adapter dir or full model)")
    parser.add_argument('--base_model_path', type=str, default=None,
                        help="Base model path (required for LoRA checkpoints, e.g. Qwen/Qwen3-4B)")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)