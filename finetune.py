#!/usr/bin/env python3
"""
Step 2: Fine-Tune Model

Uploads training data to OpenAI and creates a fine-tuning job.

Usage:
    python finetune.py
"""

import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from lib.config import (
    TRAINING_FILE,
    VALIDATION_FILE,
    MODEL_INFO_FILE,
    FINETUNING_BASE_MODEL,
    FINE_TUNING_SUFFIX,
    FINE_TUNING_HYPERPARAMETERS
)

# Load environment variables
load_dotenv()


def upload_file(client: OpenAI, filepath: str, purpose: str) -> str:
    """
    Upload a file to OpenAI.

    Args:
        client: OpenAI client instance
        filepath: Path to file to upload
        purpose: Purpose of the file ("fine-tune")

    Returns:
        File ID from OpenAI
    """
    print(f"  Uploading {filepath}...", end=" ")
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"✓ (ID: {response.id})")
    return response.id


def create_fine_tuning_job(client: OpenAI, training_file_id: str, validation_file_id: str = None) -> str:
    """
    Create a fine-tuning job.

    Args:
        client: OpenAI client instance
        training_file_id: ID of uploaded training file
        validation_file_id: ID of uploaded validation file (optional)

    Returns:
        Fine-tuning job ID
    """
    print("  Creating fine-tuning job...", end=" ")

    params = {
        "training_file": training_file_id,
        "model": FINETUNING_BASE_MODEL,
        "suffix": FINE_TUNING_SUFFIX,
    }

    if validation_file_id:
        params["validation_file"] = validation_file_id

    # Add hyperparameters if specified
    if FINE_TUNING_HYPERPARAMETERS:
        params["hyperparameters"] = FINE_TUNING_HYPERPARAMETERS

    response = client.fine_tuning.jobs.create(**params)
    print(f"✓ (Job ID: {response.id})")
    return response.id


def monitor_job(client: OpenAI, job_id: str):
    """
    Monitor a fine-tuning job until completion.

    Args:
        client: OpenAI client instance
        job_id: Fine-tuning job ID
    """
    print("\n" + "=" * 60)
    print("Monitoring fine-tuning job...")
    print("=" * 60)
    print()
    print("This may take 20-60 minutes depending on dataset size.")
    print("You can safely close this script and check status later by running:")
    print(f"  python -c \"from openai import OpenAI; print(OpenAI().fine_tuning.jobs.retrieve('{job_id}').status)\"")
    print()

    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")

            # Show additional info when available
            if job.trained_tokens:
                print(f"  Trained tokens: {job.trained_tokens:,}")

            last_status = status

        # Check if job is complete
        if status in ["succeeded", "failed", "cancelled"]:
            break

        # Wait before checking again
        time.sleep(30)

    print()
    print("=" * 60)

    if status == "succeeded":
        print("✅ Fine-tuning completed successfully!")
        print("=" * 60)
        print()
        print(f"Fine-tuned model ID: {job.fine_tuned_model}")
        print()

        # Save model info
        model_info = {
            "job_id": job_id,
            "model_id": job.fine_tuned_model,
            "base_model": FINETUNING_BASE_MODEL,
            "trained_tokens": job.trained_tokens,
            "status": status
        }

        with open(MODEL_INFO_FILE, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"Model info saved to: {MODEL_INFO_FILE}")
        print()
        print("Next step:")
        print("  python test_model.py")
        print()

    elif status == "failed":
        print("❌ Fine-tuning failed!")
        print("=" * 60)
        print()
        if job.error:
            print(f"Error: {job.error}")
        print()

    elif status == "cancelled":
        print("⚠️  Fine-tuning was cancelled")
        print("=" * 60)
        print()


def main():
    print("=" * 60)
    print("Email Writer - Step 2: Fine-Tune Model")
    print("=" * 60)
    print()

    # Check if training data exists
    if not Path(TRAINING_FILE).exists():
        print(f"Error: Training file not found: {TRAINING_FILE}")
        print()
        print("Please run prepare_data.py first:")
        print("  python prepare_data.py path/to/Sent.mbox your@gmail.com")
        print()
        return

    # Initialize OpenAI client
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print()
        print("Make sure you have set OPENAI_API_KEY in your .env file")
        print()
        return

    # Step 1: Upload files
    print("Step 1/3: Uploading files to OpenAI...")
    try:
        training_file_id = upload_file(client, TRAINING_FILE, "fine-tune")

        validation_file_id = None
        if Path(VALIDATION_FILE).exists():
            validation_file_id = upload_file(client, VALIDATION_FILE, "fine-tune")

    except Exception as e:
        print(f"\n❌ Error uploading files: {e}")
        return

    print()

    # Step 2: Create fine-tuning job
    print("Step 2/3: Starting fine-tuning job...")
    try:
        job_id = create_fine_tuning_job(client, training_file_id, validation_file_id)
    except Exception as e:
        print(f"\n❌ Error creating fine-tuning job: {e}")
        return

    print()

    # Step 3: Monitor job
    print("Step 3/3: Monitoring job progress...")
    try:
        monitor_job(client, job_id)
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted.")
        print(f"Job is still running. Job ID: {job_id}")
        print()
        print("To check status later:")
        print(f"  python -c \"from openai import OpenAI; print(OpenAI().fine_tuning.jobs.retrieve('{job_id}'))\"")
        print()
    except Exception as e:
        print(f"\n❌ Error monitoring job: {e}")
        print(f"Job ID: {job_id}")
        print()


if __name__ == "__main__":
    main()
