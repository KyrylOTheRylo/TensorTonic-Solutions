import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    answer = np.zeros(n_states)
    counts = np.zeros(n_states)

    for episode in episodes:
        returns = np.zeros(len(episode))
        G = 0.0

        for i in range(len(episode) - 1, -1, -1):
            reward = episode[i][1]
            G = reward + gamma * G
            returns[i] = G

        visited = np.zeros(n_states, dtype=bool)

        for i in range(len(episode)):
            state = episode[i][0]

            if not visited[state]:
                visited[state] = True
                answer[state] += returns[i]
                counts[state] += 1

    for state in range(n_states):
        if counts[state] > 0:
            answer[state] = answer[state] / counts[state]

    return np.round(answer, 4)