

def check_det_bbox(dictionary):
    if 'det_bboxes' in dictionary.keys():
        return True
    if 'det_bbox' in dictionary.keys():
        return True
    return False


def reformat_dict(representation_dict):
    reformatted_dict = {
        'page_number': representation_dict['page_number']['text'],
        'papacy': {
            'start_date': representation_dict['papacy']['start_date']['text'],
            'end_date': representation_dict['papacy']['end_date']['text'],
            'pope': representation_dict['papacy']['pope']['merged_text'],
        },
        'page_start_date': representation_dict['page_start_date'][
            'text'] if 'page_start_date' in representation_dict.keys() else ''
    }

    blocks = [block for block in representation_dict.keys() if
              block not in ['page_number', 'papacy', 'page_start_date']]
    for block in blocks:
        reformatted_dict[block] = {
            'date': ' '.join(t for t in representation_dict[block]['date']['text']),
            'place': ' '.join(t for t in representation_dict[block]['place']['text']),
            'content': ' '.join(t for t in representation_dict[block]['content']['text']),
        }

    return reformatted_dict
