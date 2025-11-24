#!/usr/bin/env python3
"""
Step 1: Prepare Training Data

Converts a Gmail .mbox export into training and validation datasets
for OpenAI fine-tuning.

Usage:
    python prepare_data.py path/to/Sent.mbox your.email@gmail.com
"""

import sys
import json
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from lib.email_cleaner import process_mbox
from lib.prompt_enhancer import enhance_generic_prompts
from lib.config import (
    TRAINING_FILE,
    VALIDATION_FILE,
    VALIDATION_SPLIT_RATIO,
    MIN_TRAINING_EXAMPLES,
    COST_PER_1K_TOKENS
)

# Load environment variables
load_dotenv()


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count.
    OpenAI uses ~4 characters per token on average.
    """
    return len(text) // 4


def estimate_cost(examples: list[dict], enhancement_api_calls: int = 0) -> dict:
    """
    Estimate the cost of fine-tuning based on the number of tokens.

    Args:
        examples: List of training examples
        enhancement_api_calls: Number of API calls made for prompt enhancement

    Returns:
        Dictionary with cost breakdown
    """
    total_tokens = 0

    for example in examples:
        for msg in example.get("messages", []):
            total_tokens += estimate_tokens(msg.get("content", ""))

    training_cost = (total_tokens / 1000) * COST_PER_1K_TOKENS["training"]

    # Estimate prompt enhancement cost (rough approximation)
    enhancement_cost = enhancement_api_calls * 0.02  # Rough per-batch cost

    return {
        "total_tokens": total_tokens,
        "training_cost_usd": training_cost,
        "enhancement_cost_usd": enhancement_cost,
        "total_cost_usd": training_cost + enhancement_cost
    }


def split_dataset(examples: list[dict], val_ratio: float = VALIDATION_SPLIT_RATIO, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Split dataset into training and validation sets.

    Args:
        examples: List of all examples
        val_ratio: Ratio of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training_examples, validation_examples)
    """
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Calculate split point
    total = len(shuffled)
    val_count = int(total * val_ratio)

    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]

    return train_examples, val_examples


def write_jsonl(examples: list[dict], filepath: str):
    """Write examples to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("Email Writer - Step 1: Prepare Data")
    print("=" * 60)
    print()

    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python prepare_data.py <mbox_file> <your_email>")
        print()
        print("Example:")
        print("  python prepare_data.py ~/Downloads/Sent.mbox you@gmail.com")
        print()
        sys.exit(1)

    mbox_path = sys.argv[1]
    user_email = sys.argv[2]

    # Validate inputs
    if not Path(mbox_path).exists():
        print(f"Error: File not found: {mbox_path}")
        sys.exit(1)

    print(f"📧 Processing mbox file: {mbox_path}")
    print(f"👤 Your email: {user_email}")
    print()

    # Step 1: Extract and clean emails
    print("Step 1/4: Extracting and cleaning emails...")
    try:
        dataset = process_mbox(mbox_path, user_email)
    except Exception as e:
        print(f"Error processing mbox file: {e}")
        sys.exit(1)

    print(f"  ✓ Found {len(dataset)} clean email examples")
    print()

    # Check minimum requirements
    if len(dataset) < MIN_TRAINING_EXAMPLES:
        print(f"⚠️  Warning: Only {len(dataset)} examples found.")
        print(f"   OpenAI requires at least {MIN_TRAINING_EXAMPLES} examples for fine-tuning.")
        print("   Consider exporting more emails or using a different mailbox.")
        sys.exit(1)

    # Step 2: Enhance generic prompts
    print("Step 2/4: Enhancing generic prompts...")
    try:
        client = OpenAI()
        enhanced_dataset, api_calls = enhance_generic_prompts(client, dataset, verbose=True)
    except Exception as e:
        print(f"  ⚠️  Warning: Prompt enhancement failed: {e}")
        print("  Continuing with original prompts...")
        enhanced_dataset = dataset
        api_calls = 0

    print()

    # Step 3: Split into train/validation
    print("Step 3/4: Splitting into training and validation sets...")
    train_examples, val_examples = split_dataset(enhanced_dataset)

    print(f"  ✓ Training examples: {len(train_examples)}")
    print(f"  ✓ Validation examples: {len(val_examples)}")
    print()

    # Step 4: Write files
    print("Step 4/4: Writing output files...")
    write_jsonl(train_examples, TRAINING_FILE)
    write_jsonl(val_examples, VALIDATION_FILE)

    print(f"  ✓ Training data: {TRAINING_FILE}")
    print(f"  ✓ Validation data: {VALIDATION_FILE}")
    print()

    # Cost estimation
    print("=" * 60)
    print("Cost Estimate")
    print("=" * 60)
    costs = estimate_cost(train_examples, api_calls)

    print(f"Total tokens: ~{costs['total_tokens']:,}")
    print(f"Prompt enhancement: ${costs['enhancement_cost_usd']:.3f}")
    print(f"Fine-tuning cost: ${costs['training_cost_usd']:.3f}")
    print(f"TOTAL ESTIMATED COST: ${costs['total_cost_usd']:.2f}")
    print()
    print("Note: This is an estimate. Actual costs may vary.")
    print()

    # Next steps
    print("=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print()
    print("Next step:")
    print("  python finetune.py")
    print()


if __name__ == "__main__":
    main()
