import os
import json
import logging
import requests
from openai import OpenAI
from environment.models import Action, Issue

# Configure logging for better visibility in Hugging Face Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# The judges will provide these via environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")

# UPDATED: Points directly to your Space URL by default
ENV_URL = os.getenv("ENV_URL", "https://syam-sashank-codereview-env.hf.space")

# Initialize OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def parse_llm_response(text: str) -> Action:
    """
    Parses the LLM's string output into a structured Action object.
    Handles Markdown code blocks commonly used by LLMs.
    """
    try:
        # Clean up Markdown JSON blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        data = json.loads(text.strip())
        
        # Validate items against the Issue model
        issues = [Issue(**item) for item in data]
        return Action(issues=issues, final=True)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        # Return empty list so the grader can still run (and likely give 0.0)
        return Action(issues=[], final=True)

def run_task(task_id: str) -> float:
    """
    Executes a single task: Reset -> LLM Inference -> Step -> Return Reward.
    """
    logger.info(f"--- Starting Task: {task_id} ---")
    
    # 1. Reset environment
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    reset_data = resp.json()
    
    session_id = reset_data["session_id"]
    obs = reset_data["observation"]
    
    # 2. Build the prompt
    prompt = f"""You are a professional security and code reviewer. 
Analyze the following Python code and identify all bugs, style issues, security flaws, performance anti-patterns, and missing documentation.

Return ONLY a JSON list where each item has:
- "line": (integer)
- "category": (one of: bug, style, security, performance, documentation)
- "description": (string, max 200 chars)

Code to review:
{obs['code']}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Crucial for reproducible baseline scores
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Completion error: {e}")
        raw_content = "[]"

    # Convert LLM text to Action object
    action = parse_llm_response(raw_content)

    # 3. Take step in the environment
    step_resp = requests.post(f"{ENV_URL}/step", json={
        "session_id": session_id,
        "action": action.dict()
    })
    step_resp.raise_for_status()
    result_data = step_resp.json()
    
    # Extract the F1-based reward
    final_reward = result_data["reward"]["value"]
    logger.info(f"Result for {task_id}: Score = {final_reward:.3f}")
    
    return final_reward

if __name__ == "__main__":
    # The competition requires scores for at least 3 tasks
    task_list = ["easy", "medium", "hard"]
    final_scores = {}

    print(f"Connecting to environment at: {ENV_URL}")
    
    for task in task_list:
        try:
            score = run_task(task)
            final_scores[task] = score
        except Exception as e:
            logger.error(f"Task {task} failed to execute: {e}")
            final_scores[task] = 0.0

    # Final Summary for the Logs
    print("\n" + "="*30)
    print(" BASELINE PERFORMANCE REPORT ")
    print("="*30)
    for task, score in final_scores.items():
        print(f"Task: {task:8} | Score: {score:.3f}")
    print("="*30)