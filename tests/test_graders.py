import sys
import os
# Add parent directory to path to import environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.graders import grade_easy, grade_medium, grade_hard
from environment.models import Issue

def test_grader_variance():
    # Perfect score
    perfect = [
        Issue(line=8, category="bug", description=""),
        Issue(line=2, category="documentation", description="")
    ]
    assert grade_easy(perfect) == 1.0
    
    # Empty
    assert grade_easy([]) == 0.0
    
    # Partial
    partial = [Issue(line=8, category="bug", description="")]
    # F1 score for partial match should be between 0 and 1
    # TP=1, FP=0, FN=1 -> P=1, R=0.5 -> F1 = 2*(1*0.5)/(1+0.5) = 2/3 = 0.667
    assert grade_easy(partial) == 0.667
    
    # False positive
    fp = [Issue(line=99, category="bug", description="")]
    assert grade_easy(fp) == 0.0
    
    print("All grader variance tests passed.")

if __name__ == "__main__":
    test_grader_variance()
