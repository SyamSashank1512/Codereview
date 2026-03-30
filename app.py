import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment.env import CodeReviewEnv
from environment.models import Action, Observation, Reward

app = FastAPI(title="CodeReviewEnv")

# In-memory store for environments categorized by session ID
sessions = {}

class ResetRequest(BaseModel):
    task_id: str

class StepRequest(BaseModel):
    session_id: str
    action: Action

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset_endpoint(req: ResetRequest):
    try:
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
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step_endpoint(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID or expired environment.")
    
    env = sessions[req.session_id]
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state_endpoint(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID.")
    return sessions[session_id].state()
