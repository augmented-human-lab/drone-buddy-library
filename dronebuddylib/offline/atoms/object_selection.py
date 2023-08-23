def dis(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


def select_pointed_obj(frame: list, landmarks: list, obj_result: list):
    """
    Pick out the object the finger is pointing to.

    Args:
        frame (list): the current frame
        landmarks (list): the landmarks detected by hand_detection in this frame
        obj_result (list): the bounding boxes of the objects detected in this frame

    Returns:
        list: the bounding box of the pointed object
    """
    minn = 1000000
    size = frame.shape[:2]
    ans = []
    for xywh in obj_result:
        d = dis(landmarks[8], [xywh[0] / size[1], xywh[1] / size[0]])
        rate = xywh[2] / size[1] / dis(landmarks[5], landmarks[0])
        if (rate > 25):
            continue
        if (d < minn):
            ans = xywh
            minn = d
    return ans
