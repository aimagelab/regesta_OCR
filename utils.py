import math
import numpy as np


def find_page_number(det_bboxes, rec_texts):
    candidate = rec_texts[0]
    if candidate.isdigit():
        return True, 0

    page_number_id = None
    page_right_corner_est = np.max([bbox[3] for bbox in det_bboxes]) + 10
    min_distance = math.inf
    for i, bbox in enumerate(det_bboxes):
        bbox_right_corner_x = bbox[0]
        bbox_right_corner_y = bbox[1]
        distance = math.sqrt(math.pow(page_right_corner_est - bbox_right_corner_x, 2) + bbox_right_corner_y ** 2)

        if distance < min_distance:
            page_number_id = i
            min_distance = distance

    return False, page_number_id


def find_next_number(starting_idx, rec_texts):
    index = 0
    number = None
    for j, text in enumerate(rec_texts[starting_idx:]):
        try:
            date = float(text)
        except ValueError:
            continue
        number = int(abs(date))
        index = j + 1
        break
    return index, number
