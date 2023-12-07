## Installation
This is the list of python packages that we need 
```console
conda create --name ocr python=3.8
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
```

## Inference 
Basic code to perform ocr
```
 python main.py --src_folder [SRC_FOLDER_ABS_PATH] --dst_folder [DST_FOLDER_ABS_PATH]
```
where:
```[SRC_FOLDER_ABS_PATH]``` is the path to the folder containing the page images (in .png, .jpg, or .jpeg)
```[DST_FOLDER_ABS_PATH]``` is the path to the folder where to save the output (a .json file with the heuristic-based layout analysis+ocr and a .txt file with the plain transcription)
