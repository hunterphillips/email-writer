# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Email Writer is an educational tool for fine-tuning OpenAI models to write emails in a personal style. The workflow is:

1. Extract and clean emails from Gmail .mbox exports (prepare_data.py)
2. Enhance generic prompts with AI (batched for efficiency)
3. Fine-tune a model via OpenAI's API (finetune.py)
4. Test the custom model interactively (test_model.py)

**Key Design Principle**: Simplicity and learnability over production features. Scripts are self-contained, well-commented, with no databases or complex frameworks.

## Common Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### Workflow

```bash
# Step 1: Prepare training data (5-10 minutes)
python prepare_data.py path/to/Sent.mbox your.email@gmail.com

# Step 2: Fine-tune the model (20-60 minutes, ~$1-5)
python finetune.py

# Step 3: Test your model
python test_model.py
```

### Monitoring Fine-Tuning Job

```bash
# Check status of a running job
python -c "from openai import OpenAI; print(OpenAI().fine_tuning.jobs.retrieve('JOB_ID').status)"
```

## Architecture

### Data Flow

1. **Email Extraction** (lib/email_cleaner.py)

   - Parses .mbox files using Python's mailbox library
   - Extracts clean text from multipart MIME messages
   - Handles text/plain and text/html parts
   - Removes quoted text, signatures, HTML/CSS, metadata headers

2. **Prompt Enhancement** (lib/prompt_enhancer.py)

   - Identifies generic prompts like "Write an email in your tone"
   - Uses batched OpenAI API calls (10 emails per request) for efficiency
   - Generates specific instructions based on email content
   - Example: "Write an email in your tone" → "Write an email asking about project timeline"

3. **Dataset Creation** (prepare_data.py)

   - Creates training examples in OpenAI JSONL format
   - For replies: Uses inbound email as user prompt, outbound as assistant response
   - For non-replies: Generates synthetic intents from subject/body
   - Splits 90% training / 10% validation (configurable in lib/config.py)
   - Filters unwanted content (auto-generated, meeting invites, URL-only, etc.)

4. **Fine-Tuning** (finetune.py)

   - Uploads JSONL files to OpenAI
   - Creates fine-tuning job with gpt-4o-mini-2024-07-18
   - Monitors job progress with 30-second polling
   - Saves model info to model_info.json

5. **Testing** (test_model.py)
   - Loads fine-tuned model from model_info.json
   - Provides quick test examples
   - Interactive CLI with /compare command to compare with base model

### Message Matching Logic (email_cleaner.py)

The system builds conversation pairs by matching emails:

- **Outbound emails**: Sent by user (identified by email address)
- **Inbound emails**: Indexed by Message-ID
- **Replies**: Matched using In-Reply-To header
  - If In-Reply-To exists and matches inbound Message-ID → creates reply example
  - Otherwise → generates synthetic intent prompt

### Filtering Strategy (email_cleaner.py)

Comprehensive filtering removes non-meaningful emails:

- Auto-generated messages (do not reply, automated, etc.)
- Meeting invites (Zoom, Teams, Google Meet)
- Confirmation/tracking emails (order #, tracking number)
- Code/CSS content
- URL-only emails
- Signature-only messages
- Image-only references
- Form data patterns
- Email metadata headers

### Configuration (lib/config.py)

Centralized settings:

- `FINETUNING_BASE_MODEL`: Base model for fine-tuning (default: gpt-4o-mini-2024-07-18)
- `PROMPT_ENHANCER_MODEL`: Model for prompt enhancement (default: gpt-4o-mini)
- `PROMPT_ENHANCEMENT_BATCH_SIZE`: Emails per API call (default: 10)
- `VALIDATION_SPLIT_RATIO`: Validation percentage (default: 0.1)
- `MIN_TRAINING_EXAMPLES`: Minimum required examples (default: 10)
- `GENERIC_PROMPTS`: Set of prompts to enhance
- `FINE_TUNING_HYPERPARAMETERS`: n_epochs set to "auto"

## Output Files

- `training.jsonl`: Training dataset
- `validation.jsonl`: Validation dataset
- `model_info.json`: Fine-tuned model metadata (job_id, model_id, base_model, trained_tokens, status)

## Important Notes

- **Token Estimation**: Uses ~4 characters per token for cost estimates
- **Deduplication**: Gmail stores duplicate sent messages; deduplicated by Message-ID
- **Error Handling**: User-friendly error messages, graceful degradation if prompt enhancement fails
- **Progress Display**: Clear step-by-step output with emoji indicators
- **Batched Processing**: Prompt enhancement uses batched API calls to minimize costs
- **Random Seed**: Dataset splitting uses seed=42 for reproducibility
