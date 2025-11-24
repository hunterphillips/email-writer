"""
Shared utilities and helper functions for the Streamlit UI.
"""

import json
import os
import random
import streamlit as st
from openai import OpenAI

from lib.config import VALIDATION_SPLIT_RATIO, COST_PER_1K_TOKENS


def get_api_key() -> str:
    """
    Get OpenAI API key from session state or environment.

    Returns:
        API key string or None if not found
    """
    # Check session state first
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key

    # Fall back to environment variable
    return os.getenv('OPENAI_API_KEY')


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the API key from session state or environment.

    Returns:
        OpenAI client instance

    Raises:
        ValueError if no API key is found
    """
    api_key = get_api_key()
    if not api_key:
        raise ValueError("No OpenAI API key found. Please enter your API key.")

    return OpenAI(api_key=api_key)


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            padding: 1rem 2rem;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count (4 characters per token)
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
    enhancement_cost = enhancement_api_calls * 0.02

    return {
        "total_tokens": total_tokens,
        "training_cost_usd": training_cost,
        "enhancement_cost_usd": enhancement_cost,
        "total_cost_usd": training_cost + enhancement_cost
    }


def split_dataset(examples: list[dict], val_ratio: float = VALIDATION_SPLIT_RATIO, seed: int = 42):
    """
    Split dataset into training and validation sets.

    Args:
        examples: List of all examples
        val_ratio: Ratio of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training_examples, validation_examples)
    """
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    val_count = int(total * val_ratio)

    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]

    return train_examples, val_examples


def write_jsonl(examples: list[dict], filepath: str):
    """
    Write examples to JSONL file.

    Args:
        examples: List of examples to write
        filepath: Path to output file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def upload_file_to_openai(client: OpenAI, filepath: str, purpose: str) -> str:
    """
    Upload a file to OpenAI.

    Args:
        client: OpenAI client instance
        filepath: Path to file to upload
        purpose: Purpose of the file (e.g., "fine-tune")

    Returns:
        File ID from OpenAI
    """
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    return response.id


def create_fine_tuning_job(client: OpenAI, training_file_id: str, validation_file_id: str = None,
                          base_model: str = None, suffix: str = None,
                          hyperparameters: dict = None) -> str:
    """
    Create a fine-tuning job.

    Args:
        client: OpenAI client instance
        training_file_id: ID of uploaded training file
        validation_file_id: ID of uploaded validation file (optional)
        base_model: Base model to fine-tune
        suffix: Suffix for the fine-tuned model name
        hyperparameters: Fine-tuning hyperparameters

    Returns:
        Fine-tuning job ID
    """
    params = {
        "training_file": training_file_id,
        "model": base_model,
    }

    if suffix:
        params["suffix"] = suffix

    if validation_file_id:
        params["validation_file"] = validation_file_id

    if hyperparameters:
        params["hyperparameters"] = hyperparameters

    response = client.fine_tuning.jobs.create(**params)
    return response.id


def generate_email(client: OpenAI, model_id: str, prompt: str, max_tokens: int = 500) -> str:
    """
    Generate an email using the specified model.

    Args:
        client: OpenAI client instance
        model_id: Model ID to use for generation
        prompt: User prompt
        max_tokens: Maximum tokens in response

    Returns:
        Generated email text
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content
