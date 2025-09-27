#!/usr/bin/env python3
"""
Therapist AI Chat - Simple Testing Script
Just run: python test_chat.py
"""

import os
import sys
import subprocess

# Set default environment variables
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
os.environ.setdefault("ADAPTER_DIR", "outputs/lora")

def main():
    print("🤖 Therapist AI Chat")
    print("=" * 30)
    print("Loading model...")
    print()
    
    try:
        # Run the main terminal chat
        subprocess.run([sys.executable, "src/terminal_chat.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Activated your virtual environment")
        print("2. Installed requirements: pip install -r requirements.txt")
        print("3. Trained a model: python src/train.py")

if __name__ == "__main__":
    main()
