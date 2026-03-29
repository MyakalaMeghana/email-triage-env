# Email Triage OpenEnv Environment

## Description
This project simulates a real-world email management system where an AI agent processes incoming emails.

## Motivation
Email handling is a common real-world task. This environment helps train AI to classify and respond to emails correctly.

## Action Space
- read → read normal emails
- delete → remove spam emails
- reply → respond to urgent emails

## Observation Space
- emails: list of current emails
- remaining: number of emails left

## Reward Function
- Correct action: +1.0
- Wrong action: -0.5

## Tasks

### Easy
Process at least 2 emails correctly

### Medium
Achieve higher accuracy with fewer mistakes

### Hard
Maximize reward with minimum steps

## Setup Instructions

Install dependencies:
pip install -r requirements.txt

Run project:
python inference.py

## Baseline Scores
easy: 1.0  
medium: 1.0  
hard: 1.0
