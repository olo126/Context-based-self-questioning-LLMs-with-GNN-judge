import torch, argparse, os, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

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
    # Strip any think blocks the template appended
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
    """Strip <think>...</think> blocks from generated output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def make_prompt(question):
    return (
        f"Question: {question}\n"
        "Let's think step by step.\n"
    )

def extract_last_number(text):
    if not text:
        return None
    text_clean = text.replace(',', '')
    matches = re.findall(r'-?\d+\.?\d*', text_clean)
    if matches:
        return matches[-1]
    return None

def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False

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
        # LoRA checkpoint: load base model + merge adapter
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
        # Full model checkpoint
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
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    model, tokenizer = load_model(args)
    stop_ids = get_stop_token_ids(tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name_safe = os.path.basename(args.model_path.rstrip("/")).replace('/', '_')
    record_path = os.path.join(args.output_dir, f"gsm8k_responses_{model_name_safe}.jsonl")

    correct_count = 0
    total_count = 0

    print(f"Starting evaluation (Batch size: {args.batch_size})...")

    with open(record_path, "w", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset[start:start + args.batch_size]

            questions = batch["question"]
            gold_answers_raw = batch["answer"]

            # Build prompts with chat template
            raw_prompts = [make_prompt(q) for q in questions]
            templated_prompts = [encode_prompt(tokenizer, p) for p in raw_prompts]

            enc = tokenizer(
                templated_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids,
                )

            max_input_len = enc["input_ids"].shape[1]

            for i in range(len(questions)):
                gen_ids = gen[i, max_input_len:]
                output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                output_text = strip_think(output_text)

                gold_raw = gold_answers_raw[i]
                gold_val_str = gold_raw.split("####")[-1].strip()
                gold_val = extract_last_number(gold_val_str)

                pred_val = extract_last_number(output_text)
                match = is_correct(pred_val, gold_val)

                total_count += 1
                if match:
                    correct_count += 1

                record = {
                    "question": questions[i],
                    "gold_reasoning": gold_raw,
                    "gold_value": gold_val,
                    "pred_value": pred_val,
                    "correct": match,
                    "model_response": output_text
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")

    out_path = os.path.join(args.output_dir, f"gsm8k_score_{model_name_safe}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"accuracy": accuracy, "correct": correct_count, "total": total_count}, f, indent=4)
    print(f"Score saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_path', required=True,
                        help="Path to checkpoint (LoRA adapter dir or full model)")
    parser.add_argument('--base_model_path', type=str, default=None,
                        help="Base model path (required for LoRA checkpoints, e.g. Qwen/Qwen3-4B)")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)