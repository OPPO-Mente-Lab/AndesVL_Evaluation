import os
import re
import ast
import itertools
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import pandas as pd
from PIL import Image

from .image_base import ImageBaseDataset
from ..smp import *

logger = get_logger("RUN")

"""
BBOX_FORMAT:
{
    "image": vlmeval.smp.vlm.encode_image_file_to_base64(image_path),
    "image_path": "/mnt/data/group/vlm_data/coco/images/train2014/COCO_train2014_000000000839.jpg",
    "question": "player in white",
    "answer": "[212.82, 0.0, 355.9, 132.41]"
}
Qwen2VL_BBOX_FORMAT:
{
    "image": vlmeval.smp.vlm.encode_image_file_to_base64(image_path),
    "image_path": "/mnt/data/group/vlm_data/coco/images/train2014/COCO_train2014_000000000839.jpg",
    "question": "player in white",
    "answer": "(213, 0), (355, 132)"
}
ClickPoint_FORMAT:
{
    "image": vlmeval.smp.vlm.encode_image_file_to_base64(image_path),
    "image_path": "/mnt/data/group/vlm_data/coco/images/train2014/COCO_train2014_000000000839.jpg",
    "question": "player in white",
    "answer": "(213, 0)"
}
"""

# SYSTEM_PROMPT = """You are an AI assistant. You are given a task query and a image."""  # noqa: E501
# SYSTEM_PROMPT_V2 = """You are an AI assistant. You are given a task query and a image."""  # noqa: E501

USER_INSTRUCTION = """Please provide the bounding box coordinate of the <|object_ref_start|>{description}<|object_ref_end|>"""
USER_INSTRUCTION_V2 = """Please provide the bounding box coordinate of the {description}"""

USER_INSTRUCTION_POINT = """Please provide the click point coordinate of the <|object_ref_start|>{description}<|object_ref_end|>"""

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Normalize boxes to [x_min, y_min, x_max, y_max]
    box1 = [min(box1[0], box1[2]), min(box1[1], box1[3]), 
            max(box1[0], box1[2]), max(box1[1], box1[3])]
    box2 = [min(box2[0], box2[2]), min(box2[1], box2[3]), 
            max(box2[0], box2[2]), max(box2[1], box2[3])]
    
    # Intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Intersection area (clamped to 0)
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Box areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def parse_gt_bbox(gt_response):
    BBOX_PATTERN = re.compile(r'\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]')
    match = BBOX_PATTERN.search(gt_response)
    if match:
        gt_bbox = list(map(float, match.groups()))
    else:
        gt_bbox = [0.0, 0.0, 0.0, 0.0]
    return gt_bbox


def extract_and_normalize_numbers(text):
    number_pattern = r'-?\d+(?:\.\d+)?'
    numbers = re.findall(number_pattern, text)
    if len(numbers) != 4:
        return None
    
    float_numbers = [float(x) for x in numbers]
    
    def is_integer_in_range(num):
        return (num >= 0 and num <= 1000 and num == int(num))
    
    if all(is_integer_in_range(num) for num in float_numbers):
        return [num / 1000 for num in float_numbers]
    else:
        return float_numbers


def parse_pred_response(pred_response, is_bbox=True):
    if is_bbox:
        """Parse bounding box from various formats with optimized regex and pattern tracking"""
        # Precompiled patterns
        BOX_PATTERN = re.compile(r'<box_start>\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\),\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\)<box_end>')
        Qwen2VL_BOX_PATTERN = re.compile(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)')

        box_matches = BOX_PATTERN.findall(pred_response)
        if box_matches:
            return [list(map(float, match)) for match in box_matches], "box_pattern"
        qwen2vl_matches = Qwen2VL_BOX_PATTERN.findall(pred_response)
        if qwen2vl_matches:
            return [list(map(float, match)) for match in qwen2vl_matches], "qwen2vl_pattern"
        no_match = extract_and_normalize_numbers(pred_response)
        return [[0.,0.,0.,0.] if no_match is None else no_match], "no_match"
    else:
        """Parse click points"""
        POINT_PATTERN = re.compile(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)')
        point_matches = POINT_PATTERN.findall(pred_response)
        if point_matches:
            return [list(map(float, match)) for match in point_matches]
        return [[0., 0.]]


def parse_pred_bbox(pred_response, image_path):
    image = Image.open(image_path)
    width, height = image.size

    bbox, bbox_type = parse_pred_response(pred_response, is_bbox=True)
    bbox = bbox[0]
    parsed_bbox = bbox if isinstance(bbox, list) else ast.literal_eval(bbox)
    is_normalized = all(0 <= val <= 1000 for val in parsed_bbox[:4])
    
    # Only normalize if values are in 0-1000 range
    if bbox_type == "qwen2vl_pattern" and is_normalized:
        parsed_bbox = [val / 1000 for val in parsed_bbox]
    elif bbox_type == "box_pattern":
        parsed_bbox = [val / 10000 for val in parsed_bbox]
    
    parsed_bbox = [
        parsed_bbox[0] * width,
        parsed_bbox[1] * height,
        parsed_bbox[2] * width,
        parsed_bbox[3] * height
    ]

    x1, y1, x2, y2 = parsed_bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_pred_clickpoint(pred_response, image_path):
    image = Image.open(image_path)
    width, height = image.size

    parsed_points_list = parse_pred_response(pred_response, is_bbox=False)
    parsed_points = parsed_points_list[0]  # Get the first (and likely only) point
    is_normalized = all(0 <= val <= 1000 for val in parsed_points[:2])

    if is_normalized:
        parsed_points = [val / 1000 for val in parsed_points]
    
    normed_points = [
        parsed_points[0] * width,
        parsed_points[1] * height,
    ]

    return normed_points


class RefCOCOSeries(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "Grounding"
    DATASET_URL = {
        'refcoco_val': 'refcoco_val.tsv',
        'refcoco_testA': 'refcoco_testA.tsv',
        'refcoco_testB': 'refcoco_testB.tsv',
        'refcoco+_val': 'refcoco+_val.tsv',
        'refcoco+_testA': 'refcoco+_testA.tsv',
        'refcoco+_testB': 'refcoco+_testB.tsv',
        'refcocog_val': 'refcocog_val.tsv',
        'refcocog_test': 'refcocog_test.tsv',
    }
    DATASET_MD5 = {}
    EVAL_TYPE = "rectangle"     # point or rectangle
    BBOX_TYPE = "qwen2vl"       # bbox formation

    def __init__(self, dataset="refcoco_val", skip_noimg=True, bbox_type="qwen2vl"):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        self.BBOX_TYPE = bbox_type

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        data["index"] = [str(idx + 1) for idx, x in enumerate(data["image"])]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if "image" in data:
            data["image"] = [str(x) for x in data["image"]]
            image_map = {x: y for x, y in zip(data["index"], data["image"])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data["index"]]
            data["image"] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if "img_filename" in data:
            paths = [toliststr(x) for x in data["img_filename"]]
            data["image_path"] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data["index"]]):
            data["index"] = [int(x) for x in data["index"]]

        self.data = data
        self.post_build(dataset)

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        data_path = osp.join(data_root, "RefCOCOSeries", url)
        return pd.DataFrame(load(data_path))

    def dump_image(self, line):
        assert "image_path" in line
        tgt_path = line["image_path"]
        return tgt_path

    @classmethod
    def get_action_space(self):
        return ""

    @classmethod
    def get_trajectory(self, line):
        traj_dict = {}
        traj_dict["task"] = line["question"]
        return traj_dict

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        tgt_path = self.dump_image(line)

        if self.EVAL_TYPE == "rectangle":
            if self.BBOX_TYPE == "qwen2vl":
                user_instruction = USER_INSTRUCTION.format(description=line["question"])
            else:
                user_instruction = USER_INSTRUCTION_V2.format(description=line["question"])
        else:
            user_instruction = USER_INSTRUCTION_POINT.format(description=line["question"])
        
        msgs = []
        # # add system prompt
        # if self.BBOX_TYPE == "qwen2vl":
        #     msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT))
        # else:
        #     msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT_V2))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=user_instruction))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        if self.EVAL_TYPE == "point":
            return self.evaluate_point(eval_file, **judge_kwargs)
        elif self.EVAL_TYPE == "rectangle":
            return self.evaluate_rectangle(eval_file, **judge_kwargs)

    def evaluate_rectangle(self, eval_file, **judge_kwargs):
        score_pth = eval_file.replace(".xlsx", "_score.json")
        if os.path.exists(score_pth):
            return json.load(open(score_pth))
        scorers = {
            "IoU": compute_iou,
            "ACC@0.1": lambda x, y: compute_accuracy(x, y, 0.1),
            "ACC@0.3": lambda x, y: compute_accuracy(x, y, 0.3),
            "ACC@0.5": lambda x, y: compute_accuracy(x, y, 0.5),
            "ACC@0.7": lambda x, y: compute_accuracy(x, y, 0.7),
            "ACC@0.9": lambda x, y: compute_accuracy(x, y, 0.9),
            "Center_ACC": compute_center_accuracy,
        }
        results_dict = {key: [] for key in scorers.keys()}

        result = []
        data = load(eval_file)

        assert "answer" in data and "prediction" in data
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        
        for i in tqdm(range(len(lines))):
            line = lines[i]
            gt_answer = line["answer"]
            gt_bbox = parse_gt_bbox(gt_answer)
            pred = str(line["prediction"])
            pred_bbox = parse_pred_bbox(pred, line["image_path"])
            
            try:
                match = {}
                for score_key, score_value in scorers.items():
                    score = score_value(pred_bbox, gt_bbox)
                    if score_key != "IoU":
                        match[score_key.replace("ACC", "match")] = score
                    results_dict[score_key].append(score)
            except:
                gt_bbox = None
                match = {score_key.replace("ACC", "match"): False for score_key in scorers.keys() if score_key != "IoU"}
                
            result.append({
                "img_path": line["image_path"],
                "target": line["question"],
                "gt_bbox": gt_bbox,
                "pred_bbox": pred_bbox,
                "num_matched": sum(match.values()),
                **match,
            })
            
        for key in results_dict:
            if len(results_dict[key]) == 0:
                results_dict[key] = str(0)
            else:
                results_dict[key] = str(sum(results_dict[key]) / len(results_dict[key]))
                
        score_pth = eval_file.replace(".xlsx", "_score.json")
        dump(results_dict, score_pth)

        return results_dict

    def evaluate_point(self, eval_file, **judge_kwargs):
        score_pth = eval_file.replace(".xlsx", "_score.json")
        if os.path.exists(score_pth):
            return json.load(open(score_pth))
        
        stats = defaultdict(list)
        result = []

        data = load(eval_file)
        assert "answer" in data and "prediction" in data

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            gt_answer = line["answer"]
            gt_bbox = parse_gt_bbox(gt_answer)
            pred = str(line["prediction"])

            try:
                pred_clickpoint = parse_pred_clickpoint(pred, line["image_path"])

                match = (gt_bbox[0] <= pred_clickpoint[0] <= gt_bbox[2]) and \
                        (gt_bbox[1] <= pred_clickpoint[1] <= gt_bbox[3])

                if match:
                    stats["match"].append(1)
                else:
                    stats["match"].append(0)
                is_wrong_format = False
            except Exception as e:
                logger.warning(f"Unable to get correct parsed result for {line['image_path']}")
                stats["match"].append(-1)
                match, is_wrong_format, pred_clickpoint = False, True, None
            
               
            result.append({
                "img_path": line["image_path"],
                "target": line["question"],
                "gt_bbox": gt_bbox,
                "pred_result": pred,
                "pred_point": pred_clickpoint,
                "match": match,
                "is_wrong_format": is_wrong_format
            })

        final_score_dict = {}
        # Record the number of each category
        final_score_dict.update({k + ':cnt': len(stats[k]) for k in stats})
        # Calculate the Overall stats
        full_stats = []
        for v in stats.values():
            full_stats.extend(v)
        final_score_dict['Overall_Accuracy'] = np.mean([x > 0 for x in full_stats]) * 100
        final_score_dict['Format_Err_Rate'] = np.mean([x < 0 for x in full_stats]) * 100

        score_pth = eval_file.replace(".xlsx", "_score.json")
        dump(final_score_dict, score_pth)

        failure_cases_path = os.environ.get("FAILURE_CASES_PATH", None)
        if failure_cases_path is not None:
            def click_distance(bbox, pred_clickpoint):
                x, y = pred_clickpoint
                x1, y1, x2, y2 = bbox
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                abs_shift_to_center = [abs(x - xc), abs(y - yc)]  # noqa: E501
                width_outside, height_outside = [max(0, abs_shift_to_center[0] - w / 2), max(0, abs_shift_to_center[1] - h / 2)]  # noqa: E501
                return (width_outside ** 2 + height_outside ** 2) ** 0.5  # noqa: E501

            wrong_format_result = [res for res in result if res["is_wrong_format"]]
            missed_result = [res for res in result if not res["match"] and not res["is_wrong_format"]]
            missed_result.sort(key=lambda r: click_distance(r["gt_bbox"], r["pred_point"]), reverse=True)
            failure_cases = wrong_format_result + missed_result

            with open(failure_cases_path, "w") as f:
                json.dump(failure_cases, f, indent=4, ensure_ascii=False)
        return final_score_dict