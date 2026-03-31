import os
from openai import OpenAI

from env import EmailEnv
from models import Action
from tasks import evaluate_all

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def choose_action(email):
    prompt = f"""
    You are an intelligent email assistant.

    Email: {email}

    Choose ONLY one action from:
    read / delete / reply

    Rules:
    - Spam → delete
    - Urgent → reply
    - Normal → read

    Only return the action word.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in ["read", "delete", "reply"]:
            action = "read"

    except Exception as e:
        print("LLM error, fallback used:", e)
        action = "read"

    return Action(action_type=action)


def main():
    print("Starting AI Email Triage Environment...")

    env = EmailEnv()
    state = env.reset()

    total_reward = 0
    steps = 0

    for step in range(10):
        if not state.emails:
            break

        current_email = state.emails[0]
        print(f"\nStep {step+1}")
        print("Email:", current_email)

        action = choose_action(current_email)
        print("AI Action:", action.action_type)

        state, reward, done, _ = env.step(action)

        print("Reward:", reward.score)
        print("Remaining Emails:", state.remaining)

        total_reward += reward.score
        steps += 1

        if done:
            break

    print("\nFinal Total Reward:", total_reward)

    scores = evaluate_all(total_reward, steps)

    print("\nTask Scores:")
    for k, v in scores.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
