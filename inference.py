import os
from openai import OpenAI

from env import EmailEnv
from models import Action


# ✅ Use PROVIDED environment variables
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)


def choose_action(email):
    prompt = f"""
    Email: {email}
    Choose one action: read, delete, or reply.
    Only return one word.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        action = response.choices[0].message.content.strip().lower()

    except:
        action = "read"  # fallback

    return Action(action_type=action)


def main():
    env = EmailEnv()
    state = env.reset()

    total_reward = 0
    step_count = 0

    print("[START] task=email-triage", flush=True)

    while state.emails:
        email = state.emails[0]

        action = choose_action(email)

        state, reward, done, _ = env.step(action)

        step_count += 1
        total_reward += reward.score

        print(f"[STEP] step={step_count} reward={reward.score}", flush=True)

        if done:
            break

    print(f"[END] task=email-triage score={total_reward} steps={step_count}", flush=True)


if __name__ == "__main__":
    main()
