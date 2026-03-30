import os
import json
import logging
import requests
from openai import OpenAI
from environment.models import Action, Issue

# Better logging instead of quiet failures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")  # Set this for HF Spaces

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def parse_llm_response(text: str) -> Action:
    """Parse LLM output into an Action. Expects JSON list of issues."""
    try:
        # Extract JSON from markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text.strip())
        issues = [Issue(**item) for item in data]
        return Action(issues=issues, final=True)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        # Return an empty list indicating the model failed to find issues properly
        return Action(issues=[], final=True)

def run_task(task_id: str) -> float:
    # 1. Reset environment to get initial observation and session_id
    logger.info(f"Running task: {task_id}")
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    reset_data = resp.json()
    
    session_id = reset_data["session_id"]
    obs = reset_data["observation"]
    
    # 2. Build prompt using the code from the observation
    prompt = f"""You are a code reviewer. Analyze the following Python code and list all issues (bugs, style, security, performance, documentation). 
    Return a JSON list where each item has: "line" (int), "category" (one of: bug, style, security, performance, documentation), "description" (string). 
    Example: [{{"line": 5, "category": "bug", "description": "Division by zero"}}]

Code:
{obs['code']}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Reproducibility
        )
        raw = response.choices[0].message.content
        logger.debug(f"Raw Output: {raw}")
    except Exception as e:
        logger.error(f"OpenAI completion error: {e}")
        raw = "[]"
        
    action = parse_llm_response(raw)
    
    # 3. Take step using the session_id
    step_resp = requests.post(f"{ENV_URL}/step", json={
        "session_id": session_id,
        "action": action.dict()
    })
    step_resp.raise_for_status()
    data = step_resp.json()
    
    final_reward = data["reward"]["value"]
    logger.info(f"Task {task_id}: Final Score = {final_reward:.3f}")
    return final_reward

if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        try:
            scores[task] = run_task(task)
        except Exception as e:
            logger.error(f"Task execution failed ({task}): {e}")
            scores[task] = 0.0
            
    print("\n=== Baseline Results ===")
    for task, score in scores.items():
        print(f"{task}: {score:.3f}")
