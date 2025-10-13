from ...smp import *
from ...utils import can_infer
import os
import pandas as pd
from tqdm import tqdm
import json
import re
from datetime import datetime
from dateutil import parser
import jionlp as jio
import time
from pypinyin import lazy_pinyin



FAIL_MSG = 'Failed to obtain answer via API.'

def build_kie_gpt4_prompt(line):
    prediction = str(line['prediction'])
    gt = str(line['answer'])
    prompt =f"""请分析输入的图片信息内容，对以下AB两个模型提取的关键信息进行评估:
    A模型提取的关键信息:
    {prediction}

    B模型提取的关键信息：
    {gt}

    请从以下维度进行评估:
    准确性：key和value的内容是否准确匹配图片实际情况，value只需要考虑语义、数值是否一致就可以，不需要严格相同
    完整性：图片上重要的信息有无遗漏

    请基于以上维度，给出两个模型的评估结果，评估最终结果只输出获胜方是'A'或者'B'即可，如果AB平局，则输出'NAN'


    请阐述评估过程，并最后输出：获胜方：xx
    """
    return prompt


def post_check(line):
    import re
    def get_judge_res(text):
        # 匹配 `获胜方:` 或 `获胜方：`，并提取后面的单词（包括 NAN）
        match = re.search(r"获胜方[：:]\s*([A-Za-z0-9_]+)", text)
        return match.group(1).strip() if match else None
    response = line['res']
    result = get_judge_res(response)
    return result


def kie_auxeval(model, line):
    prompt = build_kie_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        inputs = [prompt, line["image_path"]]
        res = model.generate(inputs, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def kie_acc(result_file):
    data = load(result_file)
    lt = len(data)
    total_score = 0.
    num_ties = 0
    num_wins = 0
    num_loses = 0
    for i in range(lt):
        item = data.iloc[i]
        score = post_check(item)
        print(score)
        if score != None:
            if score == 'NAN':
                num_ties += 1
            elif score == 'B':
                num_loses += 1
            elif score == 'A':
                num_wins += 1
    total_score = (num_ties+num_wins)/(num_ties+num_loses)
    res = defaultdict(list)
    res['score'].append(total_score)
    res = pd.DataFrame(res)
    return res


def normalize_text(text):
    """标准化文本：中文转拼音，英文转大写"""
    if any('\u4e00' <= c <= '\u9fff' for c in text):  # 判断是否包含中文
        return ''.join(lazy_pinyin(text)).upper()
    return text.upper()

def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def is_match(pred_value, true_value):
    pred_value = str(pred_value).replace(' ', '').replace('*', '')   # 去除无效字符
    true_value = str(true_value).replace(' ', '').replace('*', '')
    """检查预测值和真实值是否匹配（支持中英文混合）"""
    if true_value == "":    # gt为空，可能存在遮挡，则都算对
        return 1
    
    threshold = 0.5
    value = anls_compute(pred_value, true_value)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls

def parse_json(json_str):
    if not isinstance(json_str, str):
        json_str = str(json_str)
    # 1. 首先尝试直接load
    try:
        curr_json_str = json_str.replace('```json','').replace('```','')
        curr_json = json.loads(curr_json_str)
        return curr_json
    except:
        pass
    # 2. 通过正则表达式寻找，会比直接去除头尾的```json ```要灵活一点
    try:
        pattern = r'"""(.*?)"""'
        curr_json_str = json_str
        match = re.search(pattern, curr_json_str, re.DOTALL)
        curr_json_str = match.group(1).strip()
        curr_json = json.loads(curr_json_str)
        return curr_json
    except:
        pass    
    try:
        pattern = r'```json(.*?)```'
        curr_json_str = json_str
        match = re.search(pattern, curr_json_str, re.DOTALL)
        curr_json_str = match.group(1).strip()
        curr_json = json.loads(curr_json_str)
        return curr_json
    except:
        pass
    # 3. 对于某一些存在 //注释的，需要特殊处理
    try:
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        curr_json_str = json_str.replace('```','').replace('json','')
        curr_json = json.loads(curr_json_str)
        return curr_json
    except:
        pass
    return None # 解析错误


def custom_parse_time(time_str):
    try:
        # 定义常见的中文时间表达
        time_mapping = {
            '上午': 'AM',
            '下午': 'PM',
            '早上': 'AM',
            '晚上': 'PM',
            '凌晨': 'AM',
            '傍晚': 'PM',
        }
        
        # 替换中文时间表达
        for cn, en in time_mapping.items():
            if cn in time_str:
                time_str = time_str.replace(cn, en)
        
        # 尝试解析带中文的时间格式（年、月、日）
        time_str = time_str.replace('年', '-').replace('月', '-').replace('日', '')
        
        # 统一分隔符为-
        time_str = time_str.replace('.', '-').replace('/', '-')
        
        # 处理不带分隔符的纯数字日期（如20230629）
        if re.fullmatch(r'^\d{8}$', time_str):
            time_str = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}"
        
        # 处理不带分隔符的纯数字日期（如2023629）
        if re.fullmatch(r'^\d{7}$', time_str):
            time_str = f"{time_str[:4]}-{time_str[4:5]}-{time_str[5:7]}" if len(time_str[5:7]) == 2 else f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:7]}"
        
        # 尝试多种常见格式
        formats = [
            '%Y-%m-%d %p',    # 2023-06-29 PM
            '%Y-%m-%d',       # 2023-06-29
            '%Y-%m-%d %H:%M', # 2023-06-29 14:30
            '%Y-%m-%d %I:%M%p', # 2023-06-29 02:30PM
            '%Y-%m',          # 2023-06 (只到月份)
            '%Y',             # 2023 (只有年份)
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        
        # 如果以上格式都不匹配，尝试更灵活的解析
        numbers = re.findall(r'\d+', time_str)
        if len(numbers) >= 3:
            year = int(numbers[0])
            month = int(numbers[1])
            day = int(numbers[2])
            return datetime(year, month, day).strftime('%Y-%m-%d %H:%M:%S')
        elif len(numbers) == 2:
            year = int(numbers[0])
            month = int(numbers[1])
            return datetime(year, month, 1).strftime('%Y-%m-%d %H:%M:%S')
        elif len(numbers) == 1:
            year = int(numbers[0])
            return datetime(year, 1, 1).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, IndexError):
        return time_str  

def parse_time_str(time_str):
    time_str = time_str.replace(".", "-")
    # 首先尝试使用比较通用的方案
    try:
        time_str = jio.parse_time(time_str, time_base=time.time())
        return time_str['time'][0]
    except:
        pass
    # 然后针对一些边角cases使用别的
    try:
        date = parser.parse(time_str)
        date = date.strftime('%Y-%m-%d %H:%M:%S')
        return date
    except:
        pass
    date = custom_parse_time(time_str)
    return date # 无法解析就返回原始字符串

def match_time_str(time_str_pred, time_str_gt, debug_mode=False):
    
    if time_str_gt == "": return 1
    if time_str_pred == "": return 0
    if time_str_pred == time_str_gt: return 1
    try:
        time_gt = parse_time_str(time_str_gt)
        time_pred = parse_time_str(time_str_pred)
        return time_gt == time_pred
    except Exception as e:
        if debug_mode:
            print(f"{time_str_pred}x{time_str_gt} time parse failed: {e}")
        return 0

def evaluate_with_pr(data):
    error = 0
    total_correct = 0
    total_predicted_fields = 0
    total_truth_fields = 0
    parsed_total_truth_fields = 0
    parse_error_cnt = 0
    error_list = []
    log_list = []

    for idx, d in data.iterrows():
        if not isinstance(d['prediction'], str):
            d['prediction'] = str(d['prediction'])
        gt = d['answer']
        truth = json.loads(gt)
        total_truth_fields += sum(len(truth_dict) for truth_dict in truth)
        pred = parse_json(d['prediction'])
        if pred == None:
            parse_error_cnt += 1
            error_list.append('Json Decode error.')
            log_list.append('Fail')
            continue
        parsed_total_truth_fields += sum(len(truth_dict) for truth_dict in truth) # 如果解析成功，也算一个解析成功的gt个数

        if not isinstance(pred,list):
            pred = [pred]
        
        error = []
        # 检查每条数据的字段
        try:
            for i, pred_dict in enumerate(pred):
                if i >= len(truth): break
                truth_dict = truth[i]
                for key, pred_value in pred_dict.items():
                    total_predicted_fields += 1
                    match_flag = False

                    if key not in truth_dict:
                        continue
                    # 1. 对于时间，使用专门的函数
                    if 'time' in key.lower() or "date" in key.lower():
                        curr_score = match_time_str(pred_value, truth_dict[key])
                    # 2. 对于普通字符串，使用ANLS
                    else:
                        curr_score = is_match(pred_value, truth_dict[key])
                    total_correct += curr_score
                    if curr_score == 1:
                        match_flag = True
                    if not match_flag:
                        error_info = f'错误字段:{key}, pred:{pred_value}, truth:{truth_dict[key]}\n****************'
                        error.append(error_info)
            log_list.append('Success')
        except:
            log_list.append('Fail')
        error_list.append(error)
        
    
    data['log'] = log_list
    data['errors'] = error_list

    # 计算总指标
    precision = total_correct / total_predicted_fields if total_predicted_fields > 0 else 0
    recall = total_correct / total_truth_fields if total_truth_fields > 0 else 0
    recall_parsed = total_correct / parsed_total_truth_fields if parsed_total_truth_fields > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_parsed = 2 * (precision * recall_parsed) / (precision + recall_parsed) if (precision + recall_parsed) > 0 else 0
    summary = {
        "precision": f'{precision:.3f}',
        "recall": f'{recall:.3f}',
        "recall_parsed": f'{recall_parsed:.3f}',
        "f1": f'{f1:.3f}',
        "f1_parsed": f'{f1_parsed:.3f}',
        "correct": total_correct,
        "predicted_fields": total_predicted_fields,
        "truth_fields": total_truth_fields,
        "parsed_truth_fields": parsed_total_truth_fields,
        "pasre_error": f"{parse_error_cnt} / {len(data)} = {parse_error_cnt / len(data) * 100:.2f}%"
    }

    return (data, summary)