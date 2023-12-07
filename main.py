from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import sys
import json
import torch
import math
from torchvision.utils import save_image

from create_representation import create_representation
from reformat_json import reformat_dict

from mmocr.apis import MMOCRInferencer
from mmocr.utils import polygon_utils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def main():
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument('--src_folder', type=str, required=True)
    parser.add_argument('--dst_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    args.dst_folder = Path(args.dst_folder)
    args.dst_folder.mkdir(parents=True, exist_ok=True)

    src_imgs = sorted([str(img) for img in Path(args.src_folder).iterdir() if img.suffix in ['.png', '.jpg', '.jpeg']])
    print(f'Found {len(src_imgs)} images in {args.src_folder}')

    base_ocr = MMOCRInferencer(det='DBNet', rec='CRNN', device=args.device)
    processor_trocr = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    model_trocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(args.device)

    results = base_ocr(src_imgs, save_vis=True, return_vis=True)

    print('Text detection run successfully')

    for i, prediction_words in enumerate(results['predictions']):
        src_img_path = Path(src_imgs[i])
        print(f'[{i + 1}]/[{len(src_imgs)}] Processing {str(src_img_path)}')

        dst_img_path = Path(args.dst_folder) / f'{src_img_path.stem}_page_level_vis.png'
        save_image(torch.from_numpy(results['visualization'][i]).permute(2, 0, 1) / 255, dst_img_path)

        dst_json_path = Path(args.dst_folder) / f'{src_img_path.stem}.json'
        dst_json_path.parent.mkdir(parents=True, exist_ok=True)

        prediction_words = {key: list(reversed(value)) for key, value in prediction_words.items()}
        # Dump the results to a json file
        # with open(dst_json_path, 'w') as f:
        #     json.dump(prediction_words, f)

        det_words = prediction_words['det_polygons']
        prediction_words['det_bboxes'] = [polygon_utils.poly2bbox(det).astype(np.int32).tolist() for det in
                                          det_words]  # [x1, y1, x2, y2]
        image = np.array(Image.open(src_imgs[i]))

        trocr_results = {
            'rec_texts': [],
            'det_bboxes': [],
        }

        for j in range(math.ceil(len(det_words) / args.batch_size)):
            print(f'[{j + 1}]/[{math.ceil(len(det_words) / args.batch_size)}] Processing batch {j}')
            bboxes = prediction_words['det_bboxes'][j * args.batch_size:(j + 1) * args.batch_size]
            bboxes_batches = np.array(bboxes)
            bboxes_max_height = np.max(bboxes_batches[:, 3] - bboxes_batches[:, 1], axis=0)
            bboxes_max_width = np.max(bboxes_batches[:, 2] - bboxes_batches[:, 0], axis=0)

            img_cuts = []
            for bbox in bboxes:
                img_cut = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                padding = ((0, bboxes_max_height - img_cut.shape[0]), (0, bboxes_max_width - img_cut.shape[1]), (0, 0))
                img_cut_padded = np.pad(img_cut, padding, constant_values=255)
                img_cuts.append(img_cut_padded)

            pixel_values = processor_trocr(images=img_cuts, return_tensors="pt").pixel_values
            generated_ids = model_trocr.generate(pixel_values.to(args.device), max_length=200)
            generated_text = processor_trocr.batch_decode(generated_ids, skip_special_tokens=True)

            trocr_results['rec_texts'].extend(generated_text)
            trocr_results['det_bboxes'].extend(bboxes)

        dst_json_path = Path(args.dst_folder) / f'{src_img_path.stem}_trocr.json'
        dst_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Dump the results to a json file
        # with open(dst_json_path, 'w') as f:
        #     json.dump(trocr_results, f)

        representation_dict = create_representation(trocr_results)

        dst_representation_path = Path(args.dst_folder) / f'{src_img_path.stem}_trocr_representation.json'
        # Dump the results to a json file
        # with open(dst_representation_path, 'w') as f:
        #     json.dump(representation_dict, f)

        reformatted_dict = reformat_dict(representation_dict)
        dst_json_path = Path(args.dst_folder) / f'{src_img_path.stem}_trocr_reformatted.json'
        with open(dst_json_path, 'w') as f:
            json.dump(reformatted_dict, f, indent=4)

        avg_word_height = np.mean([bbox[3] - bbox[1] for bbox in prediction_words['det_bboxes']])
        prediction_lines_page_level = {key: [] for key in prediction_words}

        # lines = []
        lines_counter = 0
        current_line_avg_y = -1
        for j, bbox in enumerate(prediction_words['det_bboxes']):
            bbox_avg_y = np.mean([bbox[1], bbox[3]])
            if current_line_avg_y == -1:
                current_line_avg_y = bbox_avg_y

            if (bbox_avg_y - current_line_avg_y < 0.7 * avg_word_height) and len(
                    prediction_lines_page_level['det_bboxes']) != 0:
                prediction_lines_page_level['rec_texts'][lines_counter].append(prediction_words['rec_texts'][j])
                prediction_lines_page_level['rec_scores'][lines_counter].append(prediction_words['rec_scores'][j])
                prediction_lines_page_level['det_polygons'][lines_counter].append(prediction_words['det_polygons'][j])
                prediction_lines_page_level['det_scores'][lines_counter].append(prediction_words['det_scores'][j])
                prediction_lines_page_level['det_bboxes'][lines_counter].append(prediction_words['det_bboxes'][j])
                current_line_avg_y = (current_line_avg_y + bbox_avg_y) / 2

            else:
                if len(prediction_lines_page_level['det_bboxes']) != 0:
                    lines_counter += 1
                    current_line_avg_y = -1 if len(prediction_lines_page_level['det_bboxes']) != 0 else bbox_avg_y

                prediction_lines_page_level['rec_texts'].append([prediction_words['rec_texts'][j]])
                prediction_lines_page_level['rec_scores'].append([prediction_words['rec_scores'][j]])
                prediction_lines_page_level['det_polygons'].append([prediction_words['det_polygons'][j]])
                prediction_lines_page_level['det_scores'].append([prediction_words['det_scores'][j]])
                prediction_lines_page_level['det_bboxes'].append([prediction_words['det_bboxes'][j]])

        prediction_lines_line_level = {'det_bboxes': [], 'det_polygons': [], 'rec_texts': []}

        for j, bboxes_line in enumerate(prediction_lines_page_level['det_bboxes']):
            print(f'[{j + 1}]/[{len(prediction_lines_page_level["det_bboxes"])}] Processing line {j}')
            # Merge the bboxex of the line
            line_bbox = [min([bbox[0] for bbox in bboxes_line]),
                         min([bbox[1] for bbox in bboxes_line]),
                         max([bbox[2] for bbox in bboxes_line]),
                         max([bbox[3] for bbox in bboxes_line])]
            prediction_lines_line_level['det_bboxes'].append(line_bbox)
            prediction_lines_line_level['det_polygons'].append([polygon_utils.bbox2poly(line_bbox).tolist()])

            # Cut the line from the image
            img_cut = image[line_bbox[1]:line_bbox[3], line_bbox[0]:line_bbox[2]]  # maybe expand the bbox?
            # Run inference with TrOCR
            pixel_values = processor_trocr(images=img_cut, return_tensors="pt").pixel_values
            generated_ids = model_trocr.generate(pixel_values.to(args.device), max_length=200)
            generated_text = processor_trocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
            prediction_lines_line_level['rec_texts'].append(generated_text)

        dst_json_path = Path(args.dst_folder) / f'{src_img_path.stem}_line_level_ocr.json'
        # Dump the results to a json file
        # with open(dst_json_path, 'w') as f:
        #     json.dump(prediction_lines_line_level, f)

        # Write the results in a txt
        dst_txt_path = Path(args.dst_folder) / f'{src_img_path.stem}_line_level_ocr.txt'
        with open(dst_txt_path, 'w') as f:
            for line in prediction_lines_line_level['rec_texts']:
                f.write(line + '\n')

    sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
