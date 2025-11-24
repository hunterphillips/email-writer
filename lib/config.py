"""
Shared configuration for Email Writer
"""

# OpenAI model configuration
FINETUNING_BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Base model for fine-tuning
PROMPT_ENHANCER_MODEL = "gpt-4o-mini"  # Model used to enhance generic prompts
PROMPT_ENHANCEMENT_BATCH_SIZE = 10  # Number of prompts to process per API call

# Data preparation settings
VALIDATION_SPLIT_RATIO = 0.1  # 10% of data for validation
MIN_TRAINING_EXAMPLES = 10  # Minimum examples required by OpenAI
MAX_TRAINING_EXAMPLES = 10000  # Cap to limit training time  

# Email filtering settings
GENERIC_PROMPTS = {
    "write an email in your tone",
    "write an email in your tone.",
    "write an email with your tone",
}

# Output file names
TRAINING_FILE = "training.jsonl"
VALIDATION_FILE = "validation.jsonl"
MODEL_INFO_FILE = "model_info.json"
CURRENT_JOB_FILE = "current_finetuning_job.json"

# Fine-tuning parameters
FINE_TUNING_SUFFIX = "email-writer"  # Suffix for the fine-tuned model name
FINE_TUNING_HYPERPARAMETERS = {
    "n_epochs": "auto",  # Let OpenAI determine optimal epochs
}

# Cost estimates (approximate, as of 2024)
COST_PER_1K_TOKENS = {
    "training": 0.0070,  # $0.0070 per 1K tokens for gpt-4o-mini-2024-07-18
    "validation": 0.0080,
    "prompt_enhancement": 0.000150,  # Input cost for gpt-4o-mini
    "prompt_enhancement_output": 0.000600,  # Output cost for gpt-4o-mini
}
