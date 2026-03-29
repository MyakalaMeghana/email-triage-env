import os
from openai import OpenAI

from env import EmailEnv
from models import Action
from tasks import evaluate_all

# -----------------------------
# CONFIG (ENV VARIABLES)
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"

MAX_STEPS = 10

# Initialize client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# -----------------------------
# SIMPLE BASELINE AGENT
# -----------------------------
def choose_action(email_text):
    """
    Rule-based baseline (deterministic)
    This ensures reproducible results
    """

    email_text = email_text.lower()

    if "spam" in email_text:
        return Action(action_type="delete")
    elif "urgent" in email_text:
        return Action(action_type="reply")
    elif "meeting" in email_text:
        return Action(action_type="read")
    else:
        return Action(action_type="read")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    print("Starting Email Triage Environment...\n")

    env = EmailEnv()
    state = env.reset()

    total_reward = 0
    step_count = 0

    print("Initial State:", state, "\n")

    # Run episode
    for step in range(MAX_STEPS):
        if not state.emails:
            print("No emails left.")
            break

        current_email = state.emails[0]

        # Baseline action
        action = choose_action(current_email)

        # Step environment
        state, reward, done, _ = env.step(action)

        total_reward += reward.score
        step_count += 1

        print(f"Step {step+1}")
        print(f"Email: {current_email}")
        print(f"Action: {action.action_type}")
        print(f"Reward: {reward.score}")
        print(f"Next State: {state}")
        print("-" * 40)

        if done:
            print("Episode finished.\n")
            break

    # -----------------------------
    # FINAL RESULTS
    # -----------------------------
    print("\nFinal Total Reward:", total_reward)

    # Evaluate tasks (0.0 → 1.0)
    scores = evaluate_all(total_reward, step_count)

    print("Task Scores:")
    for task, score in scores.items():
        print(f"{task}: {score}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
