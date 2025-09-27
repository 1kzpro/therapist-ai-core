#!/usr/bin/env python3
"""
Generate training and validation datasets for PCP-AI from primary care dataset.

This script converts structured medical records into realistic conversational
training data using OpenAI API to generate natural PCP-patient dialogues.

Optimized version with:
- Batch processing with concurrent API calls
- Progress tracking with tqdm
- Incremental file writing
- Better error handling and retry logic
"""

import json
import os
import random
import argparse
from typing import Dict, List, Any, Tuple, Optional
import openai
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import threading

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

FORMAT: Return ONLY a valid JSON object with this exact structure (no other text):
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

CRITICAL: Your response must be ONLY valid JSON. Do not include any explanatory text, markdown formatting, or code blocks. Start with {{ and end with }}.

Generate a comprehensive, realistic conversation that would naturally lead to collecting all the information in the medical record. Make it sound like a real doctor-patient interaction."""

    return prompt

def extract_json_from_response(content: str) -> Optional[Dict]:
    """Extract and parse JSON from OpenAI response with multiple strategies."""
    
    # Strategy 1: Look for JSON object boundaries
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_str = content[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Look for JSON array boundaries
    start_idx = content.find('[')
    end_idx = content.rfind(']') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_str = content[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to find JSON after common prefixes
    json_markers = [
        '```json',
        '```',
        'JSON:',
        'Response:',
        'Output:',
        'Result:'
    ]
    
    for marker in json_markers:
        marker_idx = content.find(marker)
        if marker_idx != -1:
            # Look for JSON after the marker
            remaining = content[marker_idx + len(marker):]
            start_idx = remaining.find('{')
            if start_idx != -1:
                json_str = remaining[start_idx:]
                end_idx = json_str.rfind('}') + 1
                if end_idx > start_idx:
                    try:
                        return json.loads(json_str[:end_idx])
                    except json.JSONDecodeError:
                        continue
    
    # Strategy 4: Try to clean and parse the entire content
    try:
        # Remove common non-JSON prefixes/suffixes
        cleaned = content.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        if cleaned.startswith('json'):
            cleaned = cleaned[4:]
        
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def create_fallback_conversation(record: Dict, system_chat: str) -> Dict:
    """Create a basic conversation structure when JSON parsing fails."""
    
    # Extract basic info for fallback
    age = record.get('age', 'unknown')
    sex = record.get('sex', 'unknown')
    complaints = record.get('complaints', [])
    
    # Create basic conversation
    messages = [
        {"role": "system", "content": system_chat},
        {"role": "user", "content": f"Hi, I'm a {age}-year-old {sex} patient. I'm here because I've been having some health concerns."},
        {"role": "assistant", "content": "Thank you for coming in. Can you tell me more about what's been bothering you?"},
        {"role": "user", "content": f"I've been experiencing {', '.join([c['symptom'] for c in complaints[:3]]) if complaints else 'various symptoms'}."},
        {"role": "assistant", "content": "I understand. Let me ask you some questions to better understand your situation. When did these symptoms first start?"},
        {"role": "user", "content": "They started about a week ago and have been getting worse."},
        {"role": "assistant", "content": "I see. Can you describe the severity on a scale of 1-10?"},
        {"role": "user", "content": "I'd say about a 7 out of 10."},
        {"role": "assistant", "content": "Thank you. Do you have any other symptoms or concerns I should know about?"},
        {"role": "user", "content": "Not really, just the main issues I mentioned."},
        {"role": "assistant", "content": "I understand. Let me ask about your medical history. Do you have any chronic conditions?"},
        {"role": "user", "content": "No significant chronic conditions that I'm aware of."},
        {"role": "assistant", "content": "Are you currently taking any medications?"},
        {"role": "user", "content": "No, I'm not taking any medications currently."},
        {"role": "assistant", "content": "Do you have any known allergies?"},
        {"role": "user", "content": "No known allergies."},
        {"role": "assistant", "content": "Thank you for that information. Based on what you've told me, I'd like to gather some additional details about your symptoms and perform a brief assessment. <READY_TO_REPORT>"}
    ]
    
    # Create basic report JSON
    report_json = {
        "patient_id": f"P{record.get('id', '0001')}",
        "visit_date": "2025-01-14",
        "age": age,
        "sex": sex,
        "chief_complaint": complaints[0]['symptom'] if complaints else "General health concerns",
        "history_of_present_illness": "Patient reports symptoms as described",
        "past_medical_history": {"chronic_conditions": [], "surgeries": [], "hospitalizations": []},
        "medications": [],
        "allergies": [],
        "family_history": "Not significant",
        "social_history": {"smoking": "No", "alcohol": "Occasional", "occupation": "Not specified"},
        "review_of_systems": {"constitutional": "No fever, no weight loss", "cardiovascular": "No chest pain", "respiratory": "No shortness of breath"},
        "physical_examination": {"general": "Appears well", "vital_signs": "Within normal limits"},
        "assessment_and_plan": "Continue monitoring symptoms, follow up as needed"
    }
    
    return {
        "messages": messages,
        "report_json": json.dumps(report_json)
    }


def generate_conversation(client: openai.OpenAI, record: Dict, system_chat: str, max_retries: int = 3) -> Optional[Dict]:
    """Generate a conversation using OpenAI API with improved JSON parsing and fallback."""
    
    prompt = create_conversation_prompt(record, system_chat)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating realistic medical conversations. Generate natural, professional dialogues between physicians and patients. You MUST return a valid JSON object with 'messages' and 'report_json' fields. Do not include any text before or after the JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON using multiple strategies
            conversation_data = extract_json_from_response(content)
            
            if conversation_data:
                # Validate the structure
                if isinstance(conversation_data, dict):
                    if 'messages' in conversation_data and 'report_json' in conversation_data:
                        # Additional validation
                        if isinstance(conversation_data['messages'], list) and len(conversation_data['messages']) > 0:
                            return conversation_data
                        else:
                            print(f"Invalid messages structure in attempt {attempt + 1}")
                    else:
                        print(f"Missing required fields in attempt {attempt + 1}")
                else:
                    print(f"Response is not a dictionary in attempt {attempt + 1}")
            else:
                print(f"Could not extract JSON from response in attempt {attempt + 1}")
            
            # If we get here, the JSON parsing failed
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed to parse valid JSON after {max_retries} attempts")
                print(f"Response content (first 1000 chars): {content[:1000]}")
                print(f"Response content (last 500 chars): {content[-500:]}")
                print("Creating fallback conversation...")
                return create_fallback_conversation(record, system_chat)
            
            # Wait before retry
            time.sleep(2 ** attempt)
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error generating conversation after {max_retries} attempts: {e}")
                print("Creating fallback conversation...")
                return create_fallback_conversation(record, system_chat)
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def process_batch(client: openai.OpenAI, batch: List[Tuple[int, Dict]], system_chat: str, 
                  output_file: str, file_lock: threading.Lock, progress_bar: tqdm) -> None:
    """Process a batch of records and write results incrementally."""
    
    for record_idx, record in batch:
        try:
            conversation = generate_conversation(client, record, system_chat)
            if conversation:
                # Add unique ID if not present
                if 'id' not in conversation:
                    conversation['id'] = f"case-{record_idx+1:04d}"
                
                # Write to file immediately with thread safety
                with file_lock:
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(conversation) + '\n')
                
                progress_bar.update(1)
            else:
                progress_bar.set_postfix_str(f"Failed: {record_idx+1}")
                
        except Exception as e:
            print(f"Error processing record {record_idx+1}: {e}")
            progress_bar.set_postfix_str(f"Error: {record_idx+1}")
            progress_bar.update(1)


def process_dataset_batch(client: openai.OpenAI, data: List[Dict], system_chat: str, 
                         output_file: str, batch_size: int = 5, max_workers: int = 3) -> None:
    """Process dataset in batches with concurrent API calls."""
    
    # Create output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)  # Start fresh
    
    # Create batches
    batches = []
    for i in range(0, len(data), batch_size):
        batch_data = [(j, data[j]) for j in range(i, min(i + batch_size, len(data)))]
        batches.append(batch_data)
    
    # Thread safety for file writing
    file_lock = threading.Lock()
    
    # Progress bar
    progress_bar = tqdm(total=len(data), desc=f"Processing {os.path.basename(output_file)}", 
                       unit="records", ncols=100)
    
    # Process batches concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, client, batch, system_chat, 
                                   output_file, file_lock, progress_bar)
            futures.append(future)
        
        # Wait for all batches to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    progress_bar.close()


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
    parser = argparse.ArgumentParser(description='Generate training data for PCP-AI using OpenAI API (Optimized)')
    parser.add_argument('--input', default='primary_care_dataset.json', 
                       help='Input primary care dataset file')
    parser.add_argument('--output-dir', default='data', 
                       help='Output directory for train.jsonl and valid.jsonl')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Maximum number of records to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of records per batch (default: 5)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum concurrent workers (default: 3)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    print("🚀 Loading primary care dataset...")
    data = load_primary_care_dataset(args.input)
    
    if args.max_records:
        data = data[:args.max_records]
        print(f"📊 Processing {len(data)} records (limited for testing)")
    else:
        print(f"📊 Processing {len(data)} records")
    
    # Load system prompts
    print("📝 Loading system prompts...")
    system_chat, system_report = load_system_prompts()
    
    # Split dataset
    train_data, valid_data = split_dataset(data, args.train_ratio)
    print(f"📈 Split: {len(train_data)} training, {len(valid_data)} validation")
    
    # Initialize OpenAI client (required)
    try:
        client = setup_openai_client()
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Error: Could not initialize OpenAI client: {e}")
        print("This script requires OpenAI API. Please set OPENAI_API_KEY environment variable.")
        return 1
    
    # Set up output files
    train_file = os.path.join(args.output_dir, 'train.jsonl')
    valid_file = os.path.join(args.output_dir, 'valid.jsonl')
    
    print(f"\n🔄 Processing with batch size: {args.batch_size}, workers: {args.max_workers}")
    print(f"⏱️  Delay between calls: {args.delay}s")
    
    # Process training data
    print(f"\n🎯 Generating training conversations...")
    start_time = time.time()
    process_dataset_batch(client, train_data, system_chat, train_file, 
                         args.batch_size, args.max_workers)
    train_time = time.time() - start_time
    
    # Process validation data
    print(f"\n🎯 Generating validation conversations...")
    start_time = time.time()
    process_dataset_batch(client, valid_data, system_chat, valid_file, 
                         args.batch_size, args.max_workers)
    valid_time = time.time() - start_time
    
    # Count generated records
    train_count = sum(1 for _ in open(train_file)) if os.path.exists(train_file) else 0
    valid_count = sum(1 for _ in open(valid_file)) if os.path.exists(valid_file) else 0
    
    print(f"\n✅ Completed!")
    print(f"📊 Generated {train_count} training and {valid_count} validation conversations")
    print(f"⏱️  Training time: {train_time:.1f}s, Validation time: {valid_time:.1f}s")
    print(f"📁 Training file: {train_file}")
    print(f"📁 Validation file: {valid_file}")
    
    return 0

if __name__ == "__main__":
    main()
