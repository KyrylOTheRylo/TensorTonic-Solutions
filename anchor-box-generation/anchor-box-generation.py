def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    # Write code here
    a = [[0 for i in range(feature_size) ] for j in range(feature_size)]
    stride = image_size/feature_size
    for j in range(feature_size):
        for i in range(feature_size):
            a[i][j] = ((j+0.5)*stride, (i+0.5)* stride)
    w= []
    h= []
    for sc in scales:
        for asp in aspect_ratios:
            w.append(sc*asp**0.5)
            h.append(sc/(asp**0.5))
    answer = []
        # Generate all anchors at every center
    for rows in a:
        for itm in rows:
            cx, cy = itm

            for width, height in zip(w, h):
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2

                answer.append([x1, y1, x2, y2])

    return answer