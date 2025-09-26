# Training Data Generation Guide

This guide explains how to generate training and validation datasets for the PCP-AI project using the `generate_training_data.py` script.

## Overview

The script converts structured medical records from `primary_care_dataset.json` into realistic conversational training data using OpenAI API. It generates comprehensive, realistic doctor-patient conversations that follow the project's system prompts.

## Setup

### 1. Environment Setup

```bash
# Activate the correct virtual environment
source .env312/bin/activate

# Install dependencies (if not already installed)
pip install openai jsonschema
```

### 2. OpenAI API Setup (Required)

The script requires OpenAI API for realistic conversation generation:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

Generate training data with realistic conversations:

```bash
python3 generate_training_data.py
```

This will:

- Process all 50 records from `primary_care_dataset.json`
- Split into 80% training (40 records) and 20% validation (10 records)
- Generate realistic conversations using OpenAI API
- Save to `data/train.jsonl` and `data/valid.jsonl`

### Testing with Limited Records

Test the script with a small number of records:

```bash
python3 generate_training_data.py --max-records 5
```

### Custom Output Directory

```bash
python3 generate_training_data.py --output-dir outputs/training_data
```

## Command Line Options

| Option          | Description                                        | Default                     |
| --------------- | -------------------------------------------------- | --------------------------- |
| `--input`       | Input primary care dataset file                    | `primary_care_dataset.json` |
| `--output-dir`  | Output directory for train.jsonl and valid.jsonl   | `data`                      |
| `--train-ratio` | Ratio of data to use for training                  | `0.8`                       |
| `--max-records` | Maximum number of records to process (for testing) | `None` (all records)        |

## Output Format

The script generates JSONL files where each line contains:

```json
{
  "id": "case-0001",
  "messages": [
    {
      "role": "system",
      "content": "You are a primary care physician intake assistant..."
    },
    {
      "role": "user",
      "content": "Hi, I'm here because I've been having sore throat..."
    },
    {
      "role": "assistant",
      "content": "Thank you for coming in. Can you tell me more..."
    },
    // ... more conversation turns (15-25 exchanges)
    {
      "role": "assistant",
      "content": "<READY_TO_REPORT>"
    }
  ],
  "report_json": "{\"patient_id\":\"P001\",\"visit_date\":\"2025-09-14\",...}"
}
```

## Conversation Quality

The script generates realistic, comprehensive conversations that include:

- **15-25 conversation exchanges** (not short template responses)
- **Natural doctor-patient dialogue** with proper medical terminology
- **Comprehensive information gathering** covering:
  - Chief complaint details (onset, duration, severity, associated symptoms)
  - Past medical history (chronic conditions, surgeries, hospitalizations)
  - Current medications and allergies
  - Family history
  - Social history (smoking, alcohol, occupation)
  - Review of systems (constitutional, cardiovascular, respiratory, GI, etc.)
  - Any available vital signs or physical exam findings
- **Proper system prompts** from the project's `prompts/system_chat.txt`
- **Clinical reasoning** shown through follow-up questions

## Examples

### Generate Full Dataset

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Generate full dataset with realistic conversations
python3 generate_training_data.py
```

### Generate Test Dataset

```bash
# Generate 5 records for testing
python3 generate_training_data.py --max-records 5
```

### Custom Split

```bash
# Use 90% for training, 10% for validation
python3 generate_training_data.py --train-ratio 0.9
```

## Cost Estimation

When using OpenAI API:

- **GPT-4**: ~$0.06 per record (50 records = ~$3.00)
- **Processing time**: ~3-5 minutes for 50 records
- **Rate limiting**: Script includes 2-second delays between API calls

## Output Files

After running the script, you'll have:

- `data/train.jsonl` - Training conversations (40 records by default)
- `data/valid.jsonl` - Validation conversations (10 records by default)

These files are ready to use with the PCP-AI training pipeline.

## Troubleshooting

### OpenAI API Issues

If you encounter API errors:

1. Check your API key: `echo $OPENAI_API_KEY`
2. Verify you have sufficient credits
3. The script will exit if OpenAI API is not available (no fallback)

### Rate Limiting

The script includes automatic rate limiting (2 seconds between calls) when using OpenAI API. If you hit rate limits:

- Increase the delay in the script
- Use a higher-tier OpenAI plan
- Process in smaller batches

### Memory Issues

For large datasets, consider:

- Using `--max-records` to process in batches
- Processing in smaller chunks

## Features

- **Realistic conversations**: Uses OpenAI GPT-4 to generate natural doctor-patient dialogues
- **Comprehensive coverage**: Covers all aspects of a primary care intake
- **Proper system prompts**: Uses the project's actual system prompts
- **Schema compliance**: Generates JSON reports that match the project schema
- **Error handling**: Continues processing even if individual records fail
- **Rate limiting**: Built-in delays to avoid API rate limits
