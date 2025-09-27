# Training Data Generation Guide

This guide explains how to generate training and validation datasets for the PCP-AI project using the optimized `generate_training_data.py` script.

## Overview

The script converts structured medical records from `primary_care_dataset.json` into realistic conversational training data using OpenAI API. It generates comprehensive, realistic doctor-patient conversations that follow the project's system prompts.

## 🚀 **Optimized Features**

- **Batch Processing**: Concurrent API calls for 3-5x faster processing
- **Real-time Progress**: Live progress bars with ETA and status updates
- **Memory Efficient**: Incremental file writing (no memory storage)
- **Error Handling**: Retry logic with exponential backoff
- **Configurable**: Adjustable batch size, concurrency, and delays

## Setup

### 1. Environment Setup

```bash
# Activate the correct virtual environment
source .env312/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
# or manually:
pip install openai jsonschema python-dotenv tqdm
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
- Generate realistic conversations using OpenAI API with concurrent processing
- Show real-time progress bars with ETA
- Save incrementally to `data/train.jsonl` and `data/valid.jsonl`

### High-Performance Processing

For faster processing with more concurrent workers:

```bash
python3 generate_training_data.py --batch-size 10 --max-workers 5
```

### Testing with Limited Records

Test the script with a small number of records:

```bash
python3 generate_training_data.py --max-records 5
```

### Conservative Processing

For slower but safer processing (useful for rate-limited accounts):

```bash
python3 generate_training_data.py --batch-size 2 --max-workers 2 --delay 2.0
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
| `--batch-size`  | Number of records per batch                        | `5`                         |
| `--max-workers` | Maximum concurrent workers                         | `3`                         |
| `--delay`       | Delay between API calls in seconds                 | `1.0`                       |

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

### High-Performance Processing

```bash
# Fast processing with more workers and larger batches
python3 generate_training_data.py --batch-size 10 --max-workers 5
```

### Generate Test Dataset

```bash
# Generate 5 records for testing
python3 generate_training_data.py --max-records 5
```

### Conservative Processing

```bash
# Slower but safer for rate-limited accounts
python3 generate_training_data.py --batch-size 2 --max-workers 2 --delay 2.0
```

### Custom Split

```bash
# Use 90% for training, 10% for validation
python3 generate_training_data.py --train-ratio 0.9
```

## Cost Estimation

When using OpenAI API:

- **GPT-4**: ~$0.06 per record (50 records = ~$3.00)
- **Processing time**: ~1-2 minutes for 50 records (with optimization)
- **Rate limiting**: Configurable delays (default: 1 second between calls)
- **Concurrent processing**: 3-5x faster than sequential processing

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

The script includes configurable rate limiting when using OpenAI API. If you hit rate limits:

- Increase the delay: `--delay 2.0`
- Reduce concurrent workers: `--max-workers 2`
- Use smaller batches: `--batch-size 2`
- Use a higher-tier OpenAI plan

### Memory Issues

The optimized script is memory efficient with incremental file writing. For very large datasets:

- Use `--max-records` to process in chunks
- Process with smaller batch sizes
- The script no longer stores all data in memory

## Features

- **Realistic conversations**: Uses OpenAI GPT-4 to generate natural doctor-patient dialogues
- **Comprehensive coverage**: Covers all aspects of a primary care intake
- **Proper system prompts**: Uses the project's actual system prompts
- **Schema compliance**: Generates JSON reports that match the project schema
- **Batch processing**: Concurrent API calls for 3-5x faster processing
- **Real-time progress**: Live progress bars with ETA and status updates
- **Memory efficient**: Incremental file writing (no memory storage)
- **Error handling**: Retry logic with exponential backoff
- **Configurable**: Adjustable batch size, concurrency, and delays
- **Rate limiting**: Configurable delays to avoid API rate limits
