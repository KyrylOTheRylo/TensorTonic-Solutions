def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    new_values = []

    for state in range(len(values)):
        action_values = []

        for action in range(len(rewards[state])):
            action_transitions = transitions[state][action]

            expected_value = sum(
                probability * value
                for probability, value in zip(action_transitions, values)
            )

            q_value = rewards[state][action] + gamma * expected_value
            action_values.append(q_value)

        new_values.append(float(max(action_values)))

    return new_values