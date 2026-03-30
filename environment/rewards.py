from environment.models import Issue, Action, Reward
from environment.graders import compute_f1
from typing import List

def compute_reward(
    action: Action,
    ground_truth: List[Issue],
    step_count: int,
    max_steps: int
) -> Reward:
    """
    Dense reward: 
    - +0.2 per correctly identified issue (true positive)
    - -0.1 per false positive
    - -0.05 per step (encourage efficiency)
    - +0.5 bonus if all issues found and final=True
    - Final episode reward = F1 score at end if final=True, else 0
    """
    # Compute current F1 based on issues reported so far
    current_f1 = compute_f1(action.issues, ground_truth)
    
    # Per-step penalty
    step_penalty = -0.05 * step_count
    
    # True positives: count matching (line, category)
    truth_set = {(i.line, i.category) for i in ground_truth}
    agent_set = {(i.line, i.category) for i in action.issues}
    tp_count = len(truth_set & agent_set)
    fp_count = len(agent_set - truth_set)
    
    tp_reward = tp_count * 0.2
    fp_penalty = fp_count * 0.1
    
    reward_value = tp_reward - fp_penalty + step_penalty
    
    # Bonus for early completion with all issues
    all_found = (tp_count == len(ground_truth))
    if action.final and all_found:
        reward_value += 0.5
        reason = f"Final answer correct! F1={current_f1}"
    elif action.final:
        reason = f"Final answer submitted with F1={current_f1}"
        # If final but not all correct, still give F1 score as final reward
        reward_value = current_f1
    else:
        reason = f"Step {step_count}: {tp_count}/{len(ground_truth)} issues found. +{tp_reward:.2f} -{fp_penalty:.2f} -{step_penalty:.2f}"
    
    # Clip to [-1, 1] for stability
    reward_value = max(-1.0, min(1.0, reward_value))
    
    return Reward(value=reward_value, reason=reason)
