import random
from typing import Dict, Any, Tuple
from environment.models import Observation, Action, Reward
from environment.tasks import TASKS
from environment.graders import grade_easy, grade_medium, grade_hard
from environment.rewards import compute_reward

class CodeReviewEnv:
    def __init__(self, task_id: str):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}")
        self.task_id = task_id
        self._state = None
        self._step_count = 0
        self._done = False
        self._final_f1 = None
        self._ground_truth = TASKS[task_id]["ground_truth"]
        self._max_steps = TASKS[task_id]["max_steps"]
        # Use a local random instance for isolation
        self._rng = random.Random(42)

    def reset(self) -> Observation:
        self._rng.seed(42)  # Set seed on instance for each reset
        self._step_count = 0
        self._done = False
        self._final_f1 = None
        task = TASKS[self.task_id]
        self._state = {
            "code": task["code"],
            "instructions": task["instructions"],
            "issues_reported": []
        }
        return Observation(
            code=self._state["code"],
            step_count=self._step_count,
            previous_feedback="",
            done=False
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode already done. Call reset().")
        
        self._step_count += 1
        self._state["issues_reported"] = action.issues
        
        # Compute reward
        reward_obj = compute_reward(
            action=action,
            ground_truth=self._ground_truth,
            step_count=self._step_count,
            max_steps=self._max_steps
        )
        
        # Check episode termination
        done = False
        info = {}
        
        if action.final or self._step_count >= self._max_steps:
            # Grade the final attempt
            if self.task_id == "easy":
                final_score = grade_easy(action.issues)
            elif self.task_id == "medium":
                final_score = grade_medium(action.issues)
            else:
                final_score = grade_hard(action.issues)
            self._final_f1 = final_score
            done = True
            info["final_f1"] = final_score
            # Override reward: give final F1 as reward for the terminal step
            reward_obj = Reward(value=final_score, reason=f"Episode finished. F1={final_score}")
        
        self._done = done
        
        obs = Observation(
            code=self._state["code"],
            step_count=self._step_count,
            previous_feedback=reward_obj.reason,
            done=done
        )
        
        return obs, reward_obj, done, info

    def state(self) -> Dict[str, Any]:
        return self._state.copy()
