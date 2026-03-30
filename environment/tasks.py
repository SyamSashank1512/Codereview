from environment.models import Issue

# Task definitions: each has code, ground truth issues, and max steps
TASKS = {
    "easy": {
        "code": '''def calculate_average(numbers):
    """Calculate average of a list of numbers."""
    total = 0
    count = 0
    for n in numbers:
        total += n
        count += 1
    # Missing division by zero check
    return total / count
''',
        "ground_truth": [
            Issue(line=8, category="bug", description="No handling of empty list: division by zero"),
            Issue(line=2, category="documentation", description="Missing docstring param/return description")
        ],
        "max_steps": 3,
        "instructions": "Find all bugs and documentation issues in the code."
    },
    "medium": {
        "code": '''def process_user_data(users):
    results = []
    for user in users:
        if user['active'] == True:
            # Logic error: using 'name' instead of 'username'
            results.append(user['name'].upper())
    return results

def fetch_users(api_key):
    # Security issue: hardcoded API key
    return [{'name': 'Alice', 'username': 'alice123', 'active': True},
            {'name': 'Bob', 'username': 'bob456', 'active': False}]
''',
        "ground_truth": [
            Issue(line=4, category="bug", description="Logic error: should use 'username' key, not 'name'"),
            Issue(line=9, category="security", description="Hardcoded API key – expose secret"),
            Issue(line=1, category="style", description="Missing type hints and docstring")
        ],
        "max_steps": 4,
        "instructions": "Find bug, security issue, and style violation."
    },
    "hard": {
        "code": '''import threading

counter = 0

def increment():
    global counter
    for _ in range(1000):
        counter += 1  # Race condition

def start_threads():
    threads = []
    for _ in range(10):
        t = threading.Thread(target=increment)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return counter

def expensive_loop(n):
    result = []
    for i in range(n):
        # Performance anti-pattern: repeated list concatenation
        result = result + [i**2]
    return result

# Security: using eval on user input
def process_expression(expr):
    return eval(expr)
''',
        "ground_truth": [
            Issue(line=8, category="bug", description="Race condition on global counter without lock"),
            Issue(line=20, category="performance", description="O(n^2) due to list concatenation; use .append()"),
            Issue(line=27, category="security", description="eval() on user input allows arbitrary code execution"),
            Issue(line=1, category="style", description="Missing module docstring and proper structure"),
            Issue(line=13, category="documentation", description="No docstring for start_threads function")
        ],
        "max_steps": 6,
        "instructions": "Find concurrency bug, performance anti-pattern, security flaw, and documentation/style issues."
    }
}
