"""Evaluate a model's accuracy on the GSM8K test split.

Usage:
    python eval_math.py --model_name_or_path <model_name_or_path> [--num_samples N] [--batch_size B] [--max_new_tokens T]
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from leanrl.agent.single_turn import build_math_messages
from leanrl.reward.math_reward import extract_gsm8k_answer, numbers_equal


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        build_math_messages(question), tokenize=False, add_generation_prompt=True
    )


def evaluate(
    model_path: str,
    questions: list[str],
    labels: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: str = "cuda",
) -> float:
    print(f"\nLoading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompts = [build_prompt(tokenizer, q) for q in questions]
    correct = 0

    for i in tqdm(range(0, len(prompts), batch_size), desc=model_path.split("/")[-1]):
        batch_prompts = prompts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        for out, label in zip(outputs, batch_labels):
            response = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            pred = extract_gsm8k_answer(response)
            gold = extract_gsm8k_answer(label)
            if pred is not None and gold is not None and numbers_equal(pred, gold):
                correct += 1

    return correct / len(questions) if questions else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K.")
    parser.add_argument("--model_name_or_path", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of test samples (default: full test set ~1319)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading GSM8K test split ...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    questions = dataset["question"]
    labels = dataset["answer"]
    print(f"Evaluating on {len(questions)} problems.")

    accuracy = evaluate(args.model_name_or_path, questions, labels, args.batch_size, args.max_new_tokens, args.device)

    print("\n" + "=" * 50)
    print(f"Model:    {args.model_name_or_path}")
    print(f"Accuracy: {accuracy:.1%}  ({int(accuracy * len(questions))}/{len(questions)})")
    print("=" * 50)


if __name__ == "__main__":
    main()
