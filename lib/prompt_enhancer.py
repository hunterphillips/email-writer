"""
Batch prompt enhancement using OpenAI API to convert generic prompts
into context-aware, specific instructions.
"""

import re
from openai import OpenAI
from lib.config import GENERIC_PROMPTS, PROMPT_ENHANCER_MODEL, PROMPT_ENHANCEMENT_BATCH_SIZE


def generate_specific_prompts_batch(client: OpenAI, email_bodies: list[str]) -> list[str]:
    """
    Use the OpenAI API to generate specific prompts for multiple emails at once.

    Args:
        client: OpenAI client instance
        email_bodies: List of email bodies to generate prompts for

    Returns:
        List of specific prompts in the same order as input
    """
    if not email_bodies:
        return []

    # Build the batched instruction
    emails_text = ""
    for i, body in enumerate(email_bodies, 1):
        # Truncate very long emails to avoid token limits
        truncated_body = body[:1000] if len(body) > 1000 else body
        emails_text += f"\n--- Email {i} ---\n{truncated_body}\n"

    instruction = (
        "For each email below, write a *concise instruction* describing its purpose. "
        "One sentence only. No filler text. Return your answers numbered, one per line.\n\n"
        "Example format:\n"
        "1. Write an email asking for a meeting time.\n"
        "2. Write an email sharing a link with friends.\n"
        "3. Write a brief message confirming availability.\n\n"
        f"Emails:{emails_text}\n"
        "Return only the numbered instructions, one per line."
    )

    response = client.chat.completions.create(
        model=PROMPT_ENHANCER_MODEL,
        messages=[{"role": "user", "content": instruction}],
    )

    # Parse the numbered response
    result_text = response.choices[0].message.content.strip()
    lines = result_text.split('\n')

    prompts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading number and dot/colon (e.g., "1. " or "1: ")
        if line[0].isdigit():
            # Find the first non-digit, non-punctuation character
            for i, char in enumerate(line):
                if char.isalpha():
                    prompts.append(line[i:].strip())
                    break
        else:
            prompts.append(line)

    return prompts


def enhance_generic_prompts(client: OpenAI, examples: list[dict], verbose: bool = True) -> tuple[list[dict], int]:
    """
    Enhance generic prompts in training examples using batched API calls.

    Args:
        client: OpenAI client instance
        examples: List of training examples in OpenAI format
        verbose: Whether to print progress messages

    Returns:
        Tuple of (enhanced_examples, num_api_calls)
    """
    # Identify generic prompts
    generic_indices = []
    generic_bodies = []

    for idx, example in enumerate(examples):
        if "messages" not in example:
            continue

        msgs = example["messages"]
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        # Check if it's a generic prompt
        if user_msg.lower().strip() in GENERIC_PROMPTS:
            generic_indices.append(idx)
            generic_bodies.append(assistant_msg)

    if verbose:
        print(f"Found {len(generic_indices)} generic prompts to enhance")

    if not generic_indices:
        return examples, 0

    # Process generic prompts in batches
    refined_prompts = []
    num_batches = (len(generic_bodies) + PROMPT_ENHANCEMENT_BATCH_SIZE - 1) // PROMPT_ENHANCEMENT_BATCH_SIZE

    for i in range(0, len(generic_bodies), PROMPT_ENHANCEMENT_BATCH_SIZE):
        batch = generic_bodies[i:i + PROMPT_ENHANCEMENT_BATCH_SIZE]
        batch_num = i // PROMPT_ENHANCEMENT_BATCH_SIZE + 1

        if verbose:
            print(f"  Processing batch {batch_num}/{num_batches} ({len(batch)} emails)...")

        batch_prompts = generate_specific_prompts_batch(client, batch)
        refined_prompts.extend(batch_prompts)

    # Verify we got the right number of responses
    if len(refined_prompts) != len(generic_indices):
        if verbose:
            print(f"  Warning: Expected {len(generic_indices)} prompts but got {len(refined_prompts)}")
        # Pad with generic prompts if needed
        while len(refined_prompts) < len(generic_indices):
            refined_prompts.append("Write an email in your tone.")

    # Replace generic prompts with refined ones
    enhanced_examples = examples.copy()
    for idx, refined_prompt in zip(generic_indices, refined_prompts):
        example = enhanced_examples[idx]
        msgs = example["messages"]
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        enhanced_examples[idx] = {
            "messages": [
                {"role": "user", "content": refined_prompt},
                {"role": "assistant", "content": assistant_msg}
            ]
        }

    if verbose:
        print(f"  Enhanced {len(generic_indices)} prompts using {num_batches} API calls")

    return enhanced_examples, num_batches
