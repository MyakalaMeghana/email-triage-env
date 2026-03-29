from models import Observation, Action, Reward

class EmailEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.emails = [
            "urgent: client issue",
            "spam offer",
            "meeting schedule"
        ]
        self.done = False
        return self.state()

    def state(self):
        return Observation(
            emails=self.emails,
            remaining=len(self.emails)
        )

    def step(self, action: Action):
        reward = 0.0

        if not self.emails:
            self.done = True
            return self.state(), Reward(score=0), True, {}

        email = self.emails.pop(0)

        if "spam" in email and action.action_type == "delete":
            reward = 1.0
        elif "urgent" in email and action.action_type == "reply":
            reward = 1.0
        elif "meeting" in email and action.action_type == "read":
            reward = 1.0
        else:
            reward = -0.5

        done = len(self.emails) == 0

        return self.state(), Reward(score=reward), done, {}
