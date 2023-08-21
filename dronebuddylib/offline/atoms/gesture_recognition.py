def dis(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


def is_pointing(landmarks: list):
    """
    Check whether the hand is in the pointing gesture.

    Args:
        landmarks (list): the hand landmarks from hand_detection

    Returns:
        bool: whether the hand is in the pointing gesture
    """
    if (dis(landmarks[12], landmarks[0]) < dis(landmarks[9], landmarks[0])
            and dis(landmarks[16], landmarks[0]) < dis(landmarks[13], landmarks[0])
            and dis(landmarks[20], landmarks[0]) < dis(landmarks[17], landmarks[0])
            and dis(landmarks[8], landmarks[0]) > dis(landmarks[5], landmarks[0])):
        return True
    return False


def is_stop_following(landmarks: list):
    """
    Check whether the hand is in the fisted gesture.

    Args:
        landmarks (list): the hand landmarks from hand_detection

    Returns:
        bool: whether the hand is in the fisted gesture
    """
    if (dis(landmarks[12], landmarks[0]) < dis(landmarks[9], landmarks[0])
            and dis(landmarks[16], landmarks[0]) < dis(landmarks[13], landmarks[0])
            and dis(landmarks[20], landmarks[0]) < dis(landmarks[17], landmarks[0])
            and dis(landmarks[8], landmarks[0]) < dis(landmarks[5], landmarks[0])):
        return True
    return False
