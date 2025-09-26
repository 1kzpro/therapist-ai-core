#!/usr/bin/env python3
"""
Generate training and validation datasets for PCP-AI from primary care dataset.

This script converts structured medical records into realistic conversational
training data using OpenAI API to generate natural PCP-patient dialogues.
"""

import json
import os
import random
import argparse
from typing import Dict, List, Any, Tuple
import openai
from datetime import datetime
import time
from dotenv import load_dotenv

# Load system prompts
def load_system_prompts():
    """Load system prompts from the project."""
    with open('prompts/system_chat.txt', 'r') as f:
        system_chat = f.read().strip()
    
    with open('prompts/system_report.txt', 'r') as f:
        system_report = f.read().strip()
    
    return system_chat, system_report

# Set up OpenAI client
def setup_openai_client():
    """Initialize OpenAI client with API key from .env file."""
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    client = openai.OpenAI(api_key=api_key)
    return client

def load_primary_care_dataset(filepath: str) -> List[Dict]:
    """Load the primary care dataset."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_conversation_prompt(record: Dict, system_chat: str) -> str:
    """Create a prompt for OpenAI to generate realistic PCP-patient conversation."""
    
    # Extract key information for context
    age = record.get('age', 'unknown')
    sex = record.get('sex', 'unknown')
    complaints = record.get('complaints', [])
    medical_history = record.get('medical_history', {})
    status_praesens = record.get('status_praesens', {})
    vitals = {}
    
    # Extract vital signs if available
    if 'skin_and_mucosa' in record and 'temperature' in record['skin_and_mucosa']:
        vitals['temperature'] = record['skin_and_mucosa']['temperature']
    if 'cardiovascular_system' in record:
        cv = record['cardiovascular_system']
        if 'blood_pressure' in cv:
            vitals['blood_pressure'] = cv['blood_pressure']
        if 'heart_rate' in cv:
            vitals['heart_rate'] = cv['heart_rate']
    if 'respiratory_system' in record and 'respiratory_rate' in record['respiratory_system']:
        vitals['respiratory_rate'] = record['respiratory_system']['respiratory_rate']
    
    # Format complaints
    complaint_text = ""
    if complaints:
        complaint_list = [f"{c['symptom']} ({c['details']})" for c in complaints]
        complaint_text = f"Chief complaints: {', '.join(complaint_list)}"
    
    # Format medical history highlights
    history_highlights = []
    if medical_history.get('chronic_conditions'):
        history_highlights.append(f"Chronic conditions: {', '.join(medical_history['chronic_conditions'])}")
    if medical_history.get('current_medications'):
        history_highlights.append(f"Current medications: {', '.join(medical_history['current_medications'])}")
    if medical_history.get('allergies'):
        history_highlights.append(f"Allergies: {', '.join(medical_history['allergies'])}")
    if medical_history.get('family_history'):
        history_highlights.append(f"Family history: {medical_history['family_history']}")
    
    history_text = "; ".join(history_highlights) if history_highlights else "No significant medical history"
    
    # Format vitals
    vitals_text = ""
    if vitals:
        vitals_list = [f"{k}: {v}" for k, v in vitals.items()]
        vitals_text = f"Vital signs: {', '.join(vitals_list)}"
    
    prompt = f"""You are a primary care physician conducting a comprehensive intake interview. Generate a realistic, detailed conversation between you and a patient based on the following medical record information:

PATIENT INFORMATION:
- Age: {age}
- Sex: {sex}
- {complaint_text}
- Medical History: {history_text}
- {vitals_text}

CONVERSATION REQUIREMENTS:
1. Create a NATURAL, REALISTIC conversation that would typically take 10-15 minutes
2. The physician should ask about 15-25 targeted questions covering:
   - Chief complaint details (onset, duration, severity, associated symptoms)
   - Past medical history (chronic conditions, surgeries, hospitalizations)
   - Current medications and allergies
   - Family history
   - Social history (smoking, alcohol, occupation)
   - Review of systems (constitutional, cardiovascular, respiratory, GI, etc.)
   - Any available vital signs or physical exam findings
3. Patient responses should be realistic and detailed
4. Include natural follow-up questions and clarifications
5. Show the physician's clinical reasoning process
6. End with the physician saying exactly: <READY_TO_REPORT>

FORMAT: Return a JSON object with this exact structure:
{{
  "messages": [
    {{"role": "system", "content": "{system_chat}"}},
    {{"role": "user", "content": "Patient's opening statement"}},
    {{"role": "assistant", "content": "Physician's first question"}},
    {{"role": "user", "content": "Patient's detailed response"}},
    {{"role": "assistant", "content": "Follow-up question"}},
    {{"role": "user", "content": "Patient's response"}},
    // ... continue for 15-25 exchanges
    {{"role": "assistant", "content": "<READY_TO_REPORT>"}}
  ],
  "report_json": "The complete JSON report matching the schema"
}}

Generate a comprehensive, realistic conversation that would naturally lead to collecting all the information in the medical record. Make it sound like a real doctor-patient interaction."""

    return prompt

def generate_conversation(client: openai.OpenAI, record: Dict, system_chat: str) -> Dict:
    """Generate a conversation using OpenAI API."""
    
    prompt = create_conversation_prompt(record, system_chat)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at creating realistic medical conversations. Generate natural, professional dialogues between physicians and patients. Always return valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000  # Increased for longer conversations
        )
        
        # Parse the response
        content = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                conversation_data = json.loads(json_str)
                
                # Ensure we have the required structure
                if 'messages' in conversation_data and 'report_json' in conversation_data:
                    return conversation_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {content[:500]}...")
        
        # If JSON parsing fails, raise an error instead of using fallback
        raise ValueError("Failed to parse OpenAI response as valid JSON")
        
    except Exception as e:
        print(f"Error generating conversation: {e}")
        raise e  # Don't use fallback, require OpenAI API


def split_dataset(data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into training and validation sets."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def save_jsonl(data: List[Dict], filepath: str):
    """Save data as JSONL format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for i, item in enumerate(data):
            # Add unique ID if not present
            if 'id' not in item:
                item['id'] = f"case-{i+1:04d}"
            
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate training data for PCP-AI using OpenAI API')
    parser.add_argument('--input', default='primary_care_dataset.json', 
                       help='Input primary care dataset file')
    parser.add_argument('--output-dir', default='data', 
                       help='Output directory for train.jsonl and valid.jsonl')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Maximum number of records to process (for testing)')
    
    args = parser.parse_args()
    
    print("Loading primary care dataset...")
    data = load_primary_care_dataset(args.input)
    
    if args.max_records:
        data = data[:args.max_records]
        print(f"Processing {len(data)} records (limited for testing)")
    else:
        print(f"Processing {len(data)} records")
    
    # Load system prompts
    print("Loading system prompts...")
    system_chat, system_report = load_system_prompts()
    
    # Split dataset
    train_data, valid_data = split_dataset(data, args.train_ratio)
    print(f"Split: {len(train_data)} training, {len(valid_data)} validation")
    
    # Initialize OpenAI client (required)
    try:
        client = setup_openai_client()
        print("OpenAI client initialized")
    except Exception as e:
        print(f"Error: Could not initialize OpenAI client: {e}")
        print("This script requires OpenAI API. Please set OPENAI_API_KEY environment variable.")
        return 1
    
    # Process training data
    print("\nGenerating training conversations...")
    train_conversations = []
    for i, record in enumerate(train_data):
        print(f"Processing training record {i+1}/{len(train_data)}")
        
        try:
            conversation = generate_conversation(client, record, system_chat)
            train_conversations.append(conversation)
        except Exception as e:
            print(f"Error processing training record {i+1}: {e}")
            continue
        
        # Add delay to avoid rate limiting
        time.sleep(2)  # Increased delay for better rate limiting
    
    # Process validation data
    print("\nGenerating validation conversations...")
    valid_conversations = []
    for i, record in enumerate(valid_data):
        print(f"Processing validation record {i+1}/{len(valid_data)}")
        
        try:
            conversation = generate_conversation(client, record, system_chat)
            valid_conversations.append(conversation)
        except Exception as e:
            print(f"Error processing validation record {i+1}: {e}")
            continue
        
        # Add delay to avoid rate limiting
        time.sleep(2)  # Increased delay for better rate limiting
    
    # Save outputs
    train_file = os.path.join(args.output_dir, 'train.jsonl')
    valid_file = os.path.join(args.output_dir, 'valid.jsonl')
    
    print(f"\nSaving training data to {train_file}")
    save_jsonl(train_conversations, train_file)
    
    print(f"Saving validation data to {valid_file}")
    save_jsonl(valid_conversations, valid_file)
    
    print(f"\nCompleted! Generated {len(train_conversations)} training and {len(valid_conversations)} validation conversations.")
    print(f"Training file: {train_file}")
    print(f"Validation file: {valid_file}")
    
    return 0

if __name__ == "__main__":
    main()
