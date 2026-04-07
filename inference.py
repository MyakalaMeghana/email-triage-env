from env import EmailEnv
from models import Action

def choose_action(email):
    email = email.lower()

    if "urgent" in email:
        return Action(action_type="reply")
    elif "spam" in email:
        return Action(action_type="delete")
    elif "meeting" in email:
        return Action(action_type="read")
    else:
        return Action(action_type="read")


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
