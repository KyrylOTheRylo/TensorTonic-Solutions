def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    relevant_set = set(relevant)

    precision = sum(item in relevant_set for item in recommended[:k])  / k
    recall = sum(item in relevant_set for item in recommended[:k]) / len(relevant)
    return [precision, recall]