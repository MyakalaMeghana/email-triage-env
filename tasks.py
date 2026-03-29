class TaskResult:
    def __init__(self, total_reward, steps_taken):
        self.total_reward = total_reward
        self.steps_taken = steps_taken


# -----------------------------
# EASY TASK
# -----------------------------
def easy_task_grader(result: TaskResult):
    """
    Goal:
    - Correctly process at least 2 emails
    """

    if result.total_reward >= 2:
        return 1.0
    elif result.total_reward >= 1:
        return 0.5
    else:
        return 0.0


# -----------------------------
# MEDIUM TASK
# -----------------------------
def medium_task_grader(result: TaskResult):
    """
    Goal:
    - Handle emails efficiently with fewer mistakes
    """

    if result.total_reward >= 2.5:
        return 1.0
    elif result.total_reward >= 1.5:
        return 0.5
    else:
        return 0.0


# -----------------------------
# HARD TASK
# -----------------------------
def hard_task_grader(result: TaskResult):
    """
    Goal:
    - Maximize reward with minimum steps
    """

    if result.total_reward >= 3 and result.steps_taken <= 3:
        return 1.0
    elif result.total_reward >= 2:
        return 0.5
    else:
        return 0.0


# -----------------------------
# RUN ALL TASKS (HELPFUL)
# -----------------------------
def evaluate_all(total_reward, steps_taken):
    result = TaskResult(total_reward, steps_taken)

    scores = {
        "easy": easy_task_grader(result),
        "medium": medium_task_grader(result),
        "hard": hard_task_grader(result),
    }

    return scores
