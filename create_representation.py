import utils


def create_representation(ocr_results):
    even, page_number_idx = utils.find_page_number(ocr_results['det_bboxes'], ocr_results['rec_texts'])

    # Only for even pages
    representation_dict = {
        'page_number':
            {
                'text': ocr_results['rec_texts'][page_number_idx],
                'det_bbox': ocr_results['det_bboxes'][page_number_idx],
            }
    }

    papacy_start_date_idx, papacy_start_date = utils.find_next_number(
        starting_idx=1 if even else 0,
        rec_texts=ocr_results['rec_texts'])

    papacy_end_date_idx_wrt_start_date, papacy_end_date = utils.find_next_number(
        starting_idx=papacy_start_date_idx + 1 if even else papacy_start_date_idx - 1,
        rec_texts=ocr_results['rec_texts'])

    papacy_end_date_idx = papacy_start_date_idx + papacy_end_date_idx_wrt_start_date

    representation_dict['papacy'] = {
        'start_date': {
            'text': papacy_start_date,
            'det_bbox': ocr_results['det_bboxes'][papacy_start_date_idx],
        },
        'end_date': {
            'text': papacy_end_date,
            'det_bbox': ocr_results['det_bboxes'][papacy_end_date_idx],
        },
        'pope': {
            'text': ocr_results['rec_texts'][1:papacy_start_date_idx],
            'det_bboxes': ocr_results['det_bboxes'][1:papacy_start_date_idx],
            'merged_text': ' '.join(t for t in ocr_results['rec_texts'][1:papacy_start_date_idx][::-1])
        }
    }

    # search for page starting date
    y_candidate = (ocr_results['det_bboxes'][papacy_start_date_idx + 3][1] +
                   ocr_results['det_bboxes'][papacy_start_date_idx + 3][3]) / 2

    y_next = (ocr_results['det_bboxes'][papacy_start_date_idx + 4][1] +
              ocr_results['det_bboxes'][papacy_start_date_idx + 4][3]) / 2

    to_keep_start_idx = papacy_start_date_idx + 3
    if abs(y_candidate - y_next) > 15:
        # The page starts with a date
        representation_dict['page_start_date'] = {
            'text': int(abs(float(ocr_results['rec_texts'][to_keep_start_idx]))),
            'det_bbox': ocr_results['det_bboxes'][to_keep_start_idx],
        }
        to_keep_start_idx += 1

    trocr_results = {key: value[to_keep_start_idx:] for key, value in ocr_results.items()}

    # search for blocks
    x_lim_dates = 380
    blocks_dates_idx = []
    for j, bbox in enumerate(trocr_results['det_bboxes']):
        if bbox[0] < x_lim_dates:
            blocks_dates_idx.append(j)

    x_lim_places = 570
    blocks_places_idx = []
    for j, bbox in enumerate(trocr_results['det_bboxes']):
        if x_lim_dates < bbox[0] < x_lim_places:
            blocks_places_idx.append(j)

    grouped_dates = {}
    if len(blocks_dates_idx) == 0:
        # no dates detected
        blocks_dates_idx.append(0)

    for j, date_idx in enumerate(blocks_dates_idx):
        if j == 0:
            grouped_dates[j] = [date_idx]
        else:
            bbox = trocr_results['det_bboxes'][date_idx]
            prev_bbox = trocr_results['det_bboxes'][blocks_dates_idx[j - 1]]
            last_block_idx = sorted(grouped_dates.keys())[-1]
            if abs(bbox[1] - prev_bbox[1]) < 40:
                if grouped_dates.get(last_block_idx):
                    grouped_dates[last_block_idx].append(date_idx)
                else:
                    grouped_dates[last_block_idx] = [date_idx]
            else:
                grouped_dates[last_block_idx + 1] = [date_idx]

    starting_text_index = 0
    for block_start_idx in grouped_dates.keys():
        # find the complete date
        dates_indexes = grouped_dates[block_start_idx]

        places_indexes = []
        places_to_keep_start_idx = None
        for place_idx_idx, place_idx in enumerate(blocks_places_idx):
            bbox_place = trocr_results['det_bboxes'][place_idx]
            if abs(bbox_place[1] - trocr_results['det_bboxes'][dates_indexes[0]][1]) < 40:
                places_indexes.append(place_idx)
                places_to_keep_start_idx = place_idx_idx + 1

        blocks_places_idx = blocks_places_idx[places_to_keep_start_idx:]

        text_indexes = []
        for text_idx, text_bbox in enumerate(trocr_results['det_bboxes'][starting_text_index:],
                                             start=starting_text_index):
            if block_start_idx != len(grouped_dates.keys()) - 1:
                next_date_bbox = trocr_results['det_bboxes'][grouped_dates[block_start_idx + 1][0]]
                if text_bbox[3] < next_date_bbox[3]:
                    text_indexes.append(text_idx)
            else:
                text_indexes.append(text_idx)
        starting_text_index = text_indexes[-1] + 1

        representation_dict[f'block_{block_start_idx}'] = {
            'date': {
                'text': [trocr_results['rec_texts'][date_idx] for date_idx in dates_indexes],
                'det_bboxes': [trocr_results['det_bboxes'][date_idx] for date_idx in dates_indexes],
            },
            'place': {
                'text': [trocr_results['rec_texts'][place_idx] for place_idx in places_indexes],
                'det_bboxes': [trocr_results['det_bboxes'][place_idx] for place_idx in places_indexes],
            },
            'content': {
                'text': [trocr_results['rec_texts'][text_idx] for text_idx in text_indexes],
                'det_bboxes': [trocr_results['det_bboxes'][text_idx] for text_idx in text_indexes],

            }
        }

    return representation_dict
