import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import CodeReviewEnv
from environment.models import Action, Issue

def test_basic_flow():
    env = CodeReviewEnv("easy")
    obs = env.reset()
    assert "calculate_average" in obs.code
    assert obs.step_count == 0
    
    # Send a dummy action
    action = Action(issues=[Issue(line=8, category="bug", description="test")], final=False)
    obs, reward, done, info = env.step(action)
    
    assert obs.step_count == 1
    assert not done
    assert reward.value > -1.0
    
    # Finish the episode
    action = Action(issues=[Issue(line=8, category="bug", description="test")], final=True)
    obs, reward, done, info = env.step(action)
    
    assert done
    assert "final_f1" in info
    print("Environment basic flow test passed.")

if __name__ == "__main__":
    test_basic_flow()
