from ...smp import *
from ...utils import can_infer


FAIL_MSG = 'Failed to obtain answer via API.'

def build_mh_gpt4_prompt(line):
    prediction = str(line['prediction'])
    gt = str(line['answer'])
    prompt = f"""请分析输入的图片信息内容，对以下AB两个模型的图片摘要进行评估:
    A模型摘要:
    {prediction}

    B模型摘要：
    {gt}

    请从以下维度进行评估:
    准确性：摘要是否准确匹配图片实际情况，只需要考虑语义、数值是否一致就可以，不需要严格相同
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


def memory_house_auxeval(model, line):
    prompt = build_mh_gpt4_prompt(line)
    log = ''
    retry = 5
    # if post_check(line, prefetch=True):
    #     res = post_check(line, prefetch=True)
    #     return dict(log='Prefetch succeed', res=res)
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


def memory_house_acc(result_file):
    data = load(result_file)
    lt = len(data)
    total_score = 0.
    num_ties = 0
    num_wins = 0
    num_loses = 0
    for i in range(lt):
        item = data.iloc[i]
        print(item)
        score = post_check(item)
        # print(score)
        if score != None:
            if score == 'NAN':
                num_ties += 1
            elif score == 'B':
                num_loses += 1
            elif score == 'A':
                num_wins += 1
    print(num_ties)
    print(num_wins)
    print(num_loses)
    total_score = (num_ties+num_wins)/(num_ties+num_loses)
    # print(f'total_score:{total_score}')
    res = defaultdict(list)
    res['score'].append(total_score)
    res = pd.DataFrame(res)
    return res
