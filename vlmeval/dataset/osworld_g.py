import os
import re
import json
import ast
import pandas as pd
import numpy as np
from PIL import Image
from typing import Union
from collections import defaultdict
from tqdm import tqdm

from .image_base import ImageBaseDataset
from ..smp import *

logger = get_logger("RUN")

SYSTEM_PROMPT = """You are an AI assistant specialized in GUI element grounding. You are given a screenshot of a GUI interface and a task instruction. Please identify and locate the specific GUI element mentioned in the instruction."""
USER_INSTRUCTION = """Please provide the bounding box coordinate of the element described in the instruction: {instruction}"""

SYSTEM_PROMPT_V2 = """You are an AI assistant specialized in GUI element grounding. You are given a screenshot of a GUI interface and a task instruction. Please identify and locate the specific GUI element mentioned in the instruction."""
USER_INSTRUCTION_V2 = """Please provide the bounding box coordinate of the element described in the instruction: {instruction}"""


class OSWorld_G(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "Grounding"
    DATASET_URL = {
        'osworld_g': 'OSWorld-G_refined.tsv',
    }
    DATASET_MD5 = {}
    EVAL_TYPE = "point"
    RE_TYPE = "functional"
    
    def __init__(self, dataset='osworld_g', skip_noimg=True, re_type="functional"):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        self.RE_TYPE = re_type

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

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

        self.data = data

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        data_path = osp.join(data_root, "OSWorld", url)
        return pd.DataFrame(load(data_path))

    def dump_image(self, line):
        assert "image_path" in line
        tgt_path = line["image_path"]
        return tgt_path
    
    def parse_gui_types(self, gui_types_str):
        """Parse GUI types string to list"""
        try:
            if isinstance(gui_types_str, str):
                # Parse JSON-like string
                gui_types_str = gui_types_str.strip('"')
                return ast.literal_eval(gui_types_str)
            elif isinstance(gui_types_str, list):
                return gui_types_str
            else:
                return []
        except Exception as e:
            return []
    
    @classmethod
    def get_action_space(cls):
        """Get available action space for OSWorld-G"""
        return ""
    
    @classmethod
    def get_trajectory(cls, line):
        """Extract trajectory information from data line"""
        traj_dict = {}
        traj_dict["task"] = line["instruction"]
        return traj_dict
    
    def build_prompt(self, line):
        """Build prompt for grounding task"""
        if isinstance(line, int):
            line = self.data.iloc[line]
        instruction = line["instruction"]
        tgt_path = self.dump_image(line)
        
        if self.RE_TYPE == "functional":
            prompt = USER_INSTRUCTION.format(instruction=instruction)
        else:
            prompt = USER_INSTRUCTION_V2.format(instruction=instruction)
        
        msgs = []
        msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        """General evaluation method"""
        if self.EVAL_TYPE == 'point':
            return self.evaluate_point(eval_file, **judge_kwargs)
        else:
            return self.evaluate_bbox(eval_file, **judge_kwargs)
    
    def evaluate_point(self, eval_file, **judge_kwargs):
        """Evaluate point-based predictions using OSWorld-G GroundingEval logic"""
        # Check if score file already exists
        score_pth = eval_file.replace(".xlsx", "_score.json")
        if os.path.exists(score_pth):
            return json.load(open(score_pth))
        
        data = load(eval_file)
        assert 'prediction' in data, "Missing 'prediction' column in evaluation data"

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        
        # Initialize stats tracking
        stats = defaultdict(list)
        
        for i, line in enumerate(tqdm(lines, desc="Evaluating points")):
            try:
                pred = line['prediction']
                pred_coords = self.extract_coordinates_from_prediction(pred)

                gt_coords = [float(x) for x in ast.literal_eval(line['box_coordinates'])]
                
                if pred_coords is None:
                    stats['match'].append(-1)  # Format error
                else:
                    # Use OSWorld-G GroundingEval logic
                    is_correct = self._eval_grounding(
                        pred_coords, 
                        line['box_type'], 
                        gt_coords, 
                        line['image_size']
                    )
                    
                    if is_correct:
                        stats['match'].append(1)
                    else:
                        stats['match'].append(0)
                
            except Exception as e:
                stats['match'].append(-1)  # Format error
        
        # Calculate final scores
        final_score_dict = {}
        
        # Record the number of each category
        final_score_dict.update({k + ':cnt': len(stats[k]) for k in stats})
        
        # Calculate the Overall stats
        full_stats = []
        for v in stats.values():
            full_stats.extend(v)
        
        final_score_dict['Overall_Accuracy'] = np.mean([x > 0 for x in full_stats])
        final_score_dict['Format_Err_Rate'] = np.mean([x < 0 for x in full_stats])
        
        # Save score JSON file
        dump(final_score_dict, score_pth)
        
        return final_score_dict
    
    def evaluate_bbox(self, eval_file, **judge_kwargs):
        """Evaluate bounding box predictions using IoU threshold"""
        # Check if score file already exists
        score_pth = eval_file.replace(".xlsx", "_score.json")
        if os.path.exists(score_pth):
            return json.load(open(score_pth))
        
        data = load(eval_file)
        assert 'prediction' in data, "Missing 'prediction' column in evaluation data"

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        
        # Initialize stats tracking
        stats = defaultdict(list)
        threshold = judge_kwargs.get('threshold', 0.5)
        
        for i, line in enumerate(tqdm(lines, desc="Evaluating bboxes")):
            try:
                pred = line['prediction']
                gt_coords = [float(x) for x in ast.literal_eval(line['box_coordinates'])]
                
                # Extract predicted coordinates
                pred_coords = self.extract_coordinates_from_prediction(pred)
                if pred_coords is None:
                    stats['match'].append(-1)  # Format error
                    continue
                
                # Calculate IoU
                iou = self.calculate_iou(gt_coords, pred_coords)
                
                # Consider correct if IoU is above threshold
                is_correct = iou >= threshold
                
                if is_correct:
                    stats['match'].append(1)
                else:
                    stats['match'].append(0)
                    
            except Exception as e:
                stats['match'].append(-1)  # Format error
        
        # Calculate final scores
        final_score_dict = {}
        
        # Record the number of each category
        final_score_dict.update({k + ':cnt': len(stats[k]) for k in stats})
        
        # Calculate the Overall stats
        full_stats = []
        for v in stats.values():
            full_stats.extend(v)
        
        final_score_dict['Overall_Accuracy'] = np.mean([x > 0 for x in full_stats])
        final_score_dict['Format_Err_Rate'] = np.mean([x < 0 for x in full_stats])
        final_score_dict['IoU_Threshold'] = threshold
        
        # Save score JSON file
        dump(final_score_dict, score_pth)
        
        return final_score_dict
    
    def extract_coordinates_from_prediction(self, prediction):
        """Extract coordinates from model prediction - matches OSWorld-G GroundingEval logic"""
        if isinstance(prediction, str):
            # Handle pyautogui format: "pyautogui.click(x=123, y=456)"
            if "pyautogui.click" in prediction or "pyautogui.moveTo" in prediction:
                coordinates = {}
                parts = prediction.split(",")
                for part in parts:
                    if "x=" in part:
                        coordinates["x"] = float(part.split("=")[1].strip())
                    elif "y=" in part:
                        coordinates["y"] = float(part.split("=")[1].strip().rstrip(")"))
                
                if "x" in coordinates and "y" in coordinates:
                    # Return as bounding box with same point for both corners
                    return [
                        coordinates["x"],
                        coordinates["y"],
                        coordinates["x"],
                        coordinates["y"],
                    ]
                else:
                    return [0, 0, 0, 0]
            
            # Handle wait commands
            elif "wait" in prediction:
                return [-1, -1, -1, -1]
            
            # Try to extract coordinates from string patterns
            coord_patterns = [
                r'\[([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+)\]',  # [x1, y1, x2, y2]
                r'\(([0-9.-]+),\s*([0-9.-]+)\),\s*\(([0-9.-]+),\s*([0-9.-]+)\)',  # (x1, y1), (x2, y2)
                r'([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+)'  # x1, y1, x2, y2
            ]
            
            for pattern in coord_patterns:
                match = re.search(pattern, prediction)
                if match:
                    try:
                        coords = [float(match.group(i)) for i in range(1, 5)]
                        return coords
                    except ValueError:
                        continue
        
        elif isinstance(prediction, list) and len(prediction) == 4:
            try:
                return [float(x) for x in prediction]
            except ValueError:
                pass
        
        return [0, 0, 0, 0]
    
    def _eval_grounding(self, coordinate, box_type, box_coordinate, image_size):
        """
        Core evaluation logic matching OSWorld-G GroundingEval._eval method
        """
        def _is_point_in_rectangle(point, rect):
            return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

        def _is_point_in_polygon(point, polygon):
            x, y = point
            n = len(polygon) // 2
            inside = False

            j = n - 1
            for i in range(n):
                xi, yi = polygon[i * 2], polygon[i * 2 + 1]
                xj, yj = polygon[j * 2], polygon[j * 2 + 1]

                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inside = not inside
                j = i

            return inside

        # Detect if coordinates are relative (between 0 and 1)
        if all(0 <= coord <= 1 for coord in coordinate):
            # Expand the coordinate to the image width and height
            coordinate = [
                coord * image_size[i % 2] for i, coord in enumerate(coordinate)
            ]

        # Get the center point of the predicted box
        center_x = (coordinate[0] + coordinate[2]) / 2
        center_y = (coordinate[1] + coordinate[3]) / 2
        center_point = [center_x, center_y]

        if box_type == "bbox":
            # For bbox, box_coordinate contains [x, y] and box_size contains [width, height]
            # Convert to [x1, y1, x2, y2] format
            box_coordinate = [
                box_coordinate[0],
                box_coordinate[1],
                box_coordinate[0] + box_coordinate[2],  # x + width
                box_coordinate[1] + box_coordinate[3],  # y + height
            ]
            return _is_point_in_rectangle(center_point, box_coordinate)
        elif box_type == "polygon":
            return _is_point_in_polygon(center_point, box_coordinate)
        elif box_type == "refusal":
            # All center points should be negative for refusal
            return all(center_point[i] < 0 for i in range(2))
        else:
            return False

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        # Ensure boxes are in [x1, y1, x2, y2] format
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]
        
        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0