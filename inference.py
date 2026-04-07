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

    while state.emails:
        email = state.emails[0]

        action = choose_action(email)

        state, reward, done, _ = env.step(action)

        total_reward += reward.score

        if done:
            break

    print("Total Reward:", total_reward)


if __name__ == "__main__":
    main()
