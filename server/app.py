import uuid
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

# Import your environment logic
# Ensure these files exist in a folder named 'environment' with an __init__.py file
from environment.env import CodeReviewEnv
from environment.models import Action, Observation, Reward

app = FastAPI(title="CodeReviewEnv")

# In-memory store for environments categorized by session IDs
sessions: Dict[str, CodeReviewEnv] = {}

class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    session_id: str
    action: Action

# --- IMPORTANT FOR HUGGING FACE ---
# This root route tells Hugging Face your app is alive. 
# Without this, the Space may stay on "Starting..."
@app.get("/")
async def root():
    return {
        "status": "online", 
        "environment": "CodeReviewEnv", 
        "tag": "openenv",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset_endpoint(req: Optional[ResetRequest] = None):
    try:
        if req is None:
            req = ResetRequest()
        # Create a new session with a unique ID
        session_id = str(uuid.uuid4())
        env = CodeReviewEnv(req.task_id)
        obs = env.reset()
        sessions[session_id] = env
                
        # Return observation alongside the session ID
        return {
            "session_id": session_id,
            "observation": obs.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reset failed: {str(e)}")

@app.post("/step")
async def step_endpoint(req: StepRequest):
    # Check if session exists
    if req.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID or expired environment.")
    
    # Corrected Indentation: This must be outside the 'if' block
    env = sessions[req.session_id]
    
    try:
        obs, reward, done, info = env.step(req.action)
        
        # Optional: Remove session if 'done' to save memory
        if done:
            del sessions[req.session_id]
            
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Step execution failed: {str(e)}")

@app.get("/state")
async def state_endpoint(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID.")
    return sessions[session_id].state()

# Note: In Hugging Face Spaces with Docker, 
# you do NOT need 'if __name__ == "__main__": uvicorn.run(...)' 
# because it is handled by your Dockerfile CMD.

def start():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)