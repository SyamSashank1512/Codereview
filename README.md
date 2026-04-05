---
title: Codereview Env
emoji: 🔥
colorFrom: red
colorTo: green
sdk: docker
sdk_version: "1.0"
app_port: 7860
app_file: app.py
pinned: false
license: mit
short_description: OpenEnv environment for code review tasks.
tags:
  - openenv
---

# CodeReviewEnv

A realistic OpenEnv environment where an AI agent performs code review on Python code snippets.

## Real-world utility

Code review is a daily task for software engineers. Automating parts of it can save time and catch bugs early. This environment allows training agents to identify bugs, style issues, security flaws, performance problems, and documentation gaps.

## Action & Observation Spaces

- **Observation**: Contains the source code, step count, and feedback from previous step.
- **Action**: A list of issues (line number, category, description) and a `final` flag to submit.

## Tasks (Easy → Medium → Hard)

| Task   | Issues | Description |
|--------|--------|-------------|
| Easy   | 2      | Missing zero-division guard and missing docstring |
| Medium | 3      | Logic error (wrong dict key), hardcoded API key, missing type hints |
| Hard   | 5      | Race condition, O(n²) anti-pattern, eval() security hole, missing docstrings |

Graders compute F1 score based on exact (line, category) matches.

## Setup

```bash
git clone <your-space-url>
cd codereview-env
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
Baseline Inference
bash
export OPENAI_API_KEY=your_key
export ENV_URL=http://localhost:7860
python inference.py
Expected baseline scores (GPT-4o-mini):

Easy: ~0.92

Medium: ~0.78

Hard: ~0.54

Deploy to HF Spaces
Create a Space with Docker, push this repo, and set environment variables API_BASE_URL, MODEL_NAME, HF_TOKEN.