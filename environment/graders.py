from typing import List
from environment.models import Issue

def compute_f1(agent_issues: List[Issue], ground_truth: List[Issue]) -> float:
    """
    Deterministic grader: exact match on line and category.
    Returns F1 score between 0.0 and 1.0.
    """
    # Convert ground truth to set of (line, category) tuples
    truth_set = {(issue.line, issue.category) for issue in ground_truth}
    agent_set = {(issue.line, issue.category) for issue in agent_issues}

    true_positives = len(truth_set & agent_set)
    false_positives = len(agent_set - truth_set)
    false_negatives = len(truth_set - agent_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 3)

def grade_easy(agent_issues: List[Issue]) -> float:
    from environment.tasks import TASKS
    return compute_f1(agent_issues, TASKS["easy"]["ground_truth"])

def grade_medium(agent_issues: List[Issue]) -> float:
    from environment.tasks import TASKS
    return compute_f1(agent_issues, TASKS["medium"]["ground_truth"])

def grade_hard(agent_issues: List[Issue]) -> float:
    from environment.tasks import TASKS
    return compute_f1(agent_issues, TASKS["hard"]["ground_truth"])
