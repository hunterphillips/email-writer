#!/usr/bin/env python3
"""
Step 3: Test Fine-Tuned Model

Interactive CLI to test your fine-tuned email writing model.

Usage:
    python test_model.py
"""

import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from lib.config import MODEL_INFO_FILE, FINETUNING_BASE_MODEL

# Load environment variables
load_dotenv()


def load_model_info() -> dict:
    """Load the fine-tuned model info from file."""
    if not Path(MODEL_INFO_FILE).exists():
        raise FileNotFoundError(
            f"Model info file not found: {MODEL_INFO_FILE}\n"
            "Please run finetune.py first to create a fine-tuned model."
        )

    with open(MODEL_INFO_FILE, 'r') as f:
        return json.load(f)


def generate_email(client: OpenAI, model_id: str, prompt: str, max_tokens: int = 500) -> str:
    """
    Generate an email using the fine-tuned model.

    Args:
        client: OpenAI client instance
        model_id: Fine-tuned model ID
        prompt: User prompt
        max_tokens: Maximum tokens in response

    Returns:
        Generated email text
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,  # Slightly creative but consistent
    )

    return response.choices[0].message.content


def compare_with_base(client: OpenAI, fine_tuned_model: str, base_model: str, prompt: str):
    """
    Compare fine-tuned model output with base model.

    Args:
        client: OpenAI client instance
        fine_tuned_model: Fine-tuned model ID
        base_model: Base model ID
        prompt: User prompt
    """
    print("\n" + "=" * 60)
    print("Comparison: Fine-Tuned vs Base Model")
    print("=" * 60)
    print()

    print(f"Prompt: {prompt}")
    print()

    print("Fine-Tuned Model:")
    print("-" * 60)
    fine_tuned_response = generate_email(client, fine_tuned_model, prompt)
    print(fine_tuned_response)
    print()

    print("Base Model:")
    print("-" * 60)
    base_response = generate_email(client, base_model, prompt)
    print(base_response)
    print()


def interactive_mode(client: OpenAI, model_id: str, base_model: str):
    """
    Run interactive testing mode.

    Args:
        client: OpenAI client instance
        model_id: Fine-tuned model ID
        base_model: Base model ID
    """
    print("\n" + "=" * 60)
    print("Interactive Testing Mode")
    print("=" * 60)
    print()
    print("Enter prompts to test your fine-tuned model.")
    print("Commands:")
    print("  /compare - Compare with base model for the last prompt")
    print("  /quit    - Exit")
    print()

    last_prompt = None

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if not prompt:
                continue

            if prompt == "/quit":
                print("\nGoodbye!")
                break

            if prompt == "/compare":
                if last_prompt:
                    compare_with_base(client, model_id, base_model, last_prompt)
                else:
                    print("No previous prompt to compare. Please enter a prompt first.")
                continue

            # Generate email
            print("\nGenerating email...")
            print("-" * 60)
            response = generate_email(client, model_id, prompt)
            print(response)
            print()

            last_prompt = prompt

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    print("=" * 60)
    print("Email Writer - Step 3: Test Model")
    print("=" * 60)
    print()

    # Load model info
    try:
        model_info = load_model_info()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    model_id = model_info.get("model_id")
    base_model = model_info.get("base_model", FINETUNING_BASE_MODEL)

    print(f"Fine-tuned model: {model_id}")
    print(f"Base model: {base_model}")
    print()

    # Initialize OpenAI client
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print()
        print("Make sure you have set OPENAI_API_KEY in your .env file")
        print()
        return

    # Run quick test examples
    print("Quick Test Examples:")
    print("=" * 60)
    print()

    test_prompts = [
        "Write a brief email asking a colleague if they're free for coffee next week.",
        "Write an email thanking someone for their help on a project.",
        "Write a short message declining a meeting invitation politely.",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Example {i}: {prompt}")
        print("-" * 60)
        try:
            response = generate_email(client, model_id, prompt, max_tokens=300)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()

    # Enter interactive mode
    try:
        interactive_mode(client, model_id, base_model)
    except Exception as e:
        print(f"Error in interactive mode: {e}")


if __name__ == "__main__":
    main()
