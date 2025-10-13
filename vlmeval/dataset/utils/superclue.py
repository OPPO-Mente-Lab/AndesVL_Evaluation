system_prompt = """# 评价标准
| 标准名称          | 标准等级 | 标准含义                                                                 |
| ----------------- | -------- | ------------------------------------------------------------------------ |
| 正确性            | 基础     | 模型能否正确的回答用户的问题。                                             |
| 相关性            | 基础     | 模型回答的内容是否与图片高度相关。                                          |
| 流畅性            | 基础     | 模型输出是否语言表达通顺，句意连贯。                                        |
| 知识延伸          | 扩展     | 模型能否在正确回答用户问题的基础上，能够给出与主题相关的扩展知识。            |
| 输出样式多样化    | 扩展     | 模型能够根据任务类型特点，自主的选择合适的方式组织输出结果，如表格呈现、时间线呈现等。 |
| 多感官信息融合    | 扩展     | 模型能够同时处理和融合来自不同感官（不同感官在本题中体现为模型是否正确的利用中文描述和图片的内容）的信息。 |

# 评价
## 基础维度：
1. **正确性**：答案正确地计算出了总花费为85.70元，这与票据上显示的信息一致。因此，可以给满分5分。（得分：5/5）
2. **相关性**：答案详细列出了购买的各种商品及其价格，与图片上的票据信息高度相关，给出5分。（得分：5/5）
3. **流畅性**：答案的语言表达较为通顺，信息列出清晰，但在部分商品的描述中出现了轻微的拼写错误（如“伊犁”应为“伊利”、“酸牛奶”应为“酸奶”），影响了部分语句的流畅性，因此给出4分。（得分：4/5）
## 扩展维度：
1. **知识延伸**：该答案仅仅列出了商品及其价格，并计算了总金额，未对用户的购物行为或者习惯进行分析或提供额外的相关信息，知识延伸较为有限。（得分：2/5）
2. **输出样式多样化**：答案以简单的列表形式列出了商品和价格，这种呈现方式较为基本，没有使用更具创意的输出样式如表格或图表，因此给出3分。（得分：3/5）
3. **多感官信息融合**：答案正确地使用了图片中的文本信息，虽然没有对文本与图片内容之间的联系进行额外解释或深入挖掘，但基本融合了视觉与文本信息，因此给出5分。（得分：5/5）

# 总结
```json
{
    '正确性': 5,
    '相关性': 5,
    '流畅性': 4,
    '知识延伸': 2,
    '输出样式多样化': 3,
    '多感官信息融合': 5
}
```

# 回复要求
用户会给你提供：一个图片，一个AI回答，一个参考答案。
请你扮演一个专业评测人员，对该多模态AI的VQA能力进行评测，参考以上评价标准中各维度的描述以及参考答案，给出你的评价结果。
回复需要包含**评价**和**总结**两部分，注意需要给出每个维度的具体评分和评价理由，评分为1-5分的正整数，5分为满分，1分为最低分，以及最后给出json格式的总结。"""

import re
import json
import pandas as pd
import requests
import json
import time
import hmac
import hashlib
import sys
sys.path.append("/mnt/data/group/wangnan/code/tools")
from thread_parallel_apply import parallel_apply
import os
from collections import OrderedDict
from PIL import Image
from io import BytesIO
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from tqdm import tqdm
import re
from openai import OpenAI

openai_host = "vllm"
if openai_host == "vllm":
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
elif openai_host=="sglang":
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:30000/v1"
elif openai_host == "lmdeploy":
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:23333/v1"
elif openai_host == "openai":
    app_id = "RongZhiLab"
    app_secret = "3qeBDOdPLgrOUPU7NE3IAMdOxQ6E1Ksh4oMmqMPwhTI="
    openai_api_key = app_id + "/" + app_secret
    openai_api_base = "https://https://andesgpt-gateway.oppoer.me/converter/openai/v1"
elif openai_host == "qwenvl":
    openai_api_key = "sk-11fc5ee16efa4e89a2bd618e9a3ce49c"
    openai_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
else:
    raise Exception()
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def sign(params, body, app_id, secret_key):
    # 1. 构建认证字符串前缀，格式为 bot-auth-v1/{appId}/{timestamp}, timestamp为时间戳，精确到毫秒，用以验证请求是否失效
    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    sb = [auth_string_prefix]
    # 2. 构建url参数字符串，按照参数名字典序升序排列
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        sb.extend(["{}={}&".format(k, v) for k, v in ordered_params.items()])
    # 3. 拼接签名原文字符串
    sign_str = "".join(sb) + body
    # 4. hmac_sha_256算法签名
    signature = hmac.new(
        secret_key.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    # 5. 拼接认证字符串
    return auth_string_prefix + signature

def get_base64(img_path):
    # 打开图片
    img = Image.open(img_path).convert("RGB")
    # 创建一个内存缓冲区
    img_buffer = BytesIO()
    # 将图片保存为PNG格式到内存缓冲区
    img.save(img_buffer, format='PNG')
    # 获取内存缓冲区中的图片数据
    img_bytes = img_buffer.getvalue()
    # 将图片数据转换为Base64编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    # 返回数据，格式固定为PNG
    return f"data:image/png;base64,{img_base64}"
def extract_score(response):
    try:
        return json.loads(re.search("```json\n(.*)```", response.text).group(1))
    except Exception as e:
        print(e)

def generate(instruction, img_path):
    req_id = "xxx"
    for i in range(10):#尝试10次
        try:
            ak = "RongZhiLab"
            sk = "3qeBDOdPLgrOUPU7NE3IAMdOxQ6E1Ksh4oMmqMPwhTI="
            body = {
                "maxTokens": 4096, #需要指定，否则会被截断
                "model": "gpt-4o", 
                "temperature": 0, #取值范围为0.0-1.0，值越大，生成的文本越随机 
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": instruction,
                        "images": [
                            {
                                "url": get_base64(img_path),
                                "detail": "high", #取值为：low/high/auto
                            }
                        ],
                    }
                ],
            }
            data = json.dumps(body)
            header = {
                "recordId": str(req_id),
                "Authorization": sign(None, data, ak, sk),
                "Content-Type": "application/json",
            }
            resp = requests.request(
                "POST",
                url="https://https://andesgpt-gateway.oppoer.me/chat/v1/completions",
                headers=header,
                data=data,
            )
            # return json.loads(resp.text)
            resp_dict = json.loads(resp.text)
            return(resp_dict["data"]["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Error:{e}")
            print(f"retry {i+1} times")
            #time.sleep((i+1)*2+5)
            time.sleep((i+1))
    return "请求失败"

def parse_score(score):
    try:
        if "```json" in score:
            score = re.findall(r"```json\n(.*?)\n```",score,re.S)[0]
        try:
            score = json.loads(score)
        except:
            score = eval(score)
        return score
    except:
        return score

def generate_response(instrution, system_prompt="", images=[]):
    openai_host = 'vllm'
    if openai_host == "openai":
        model = "gpt-4o"
    elif openai_host == "qwenvl":
        model = "qwen-vl-max"
    else:
        # 本地服务
        model = client.models.list().data[0].id
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if images:
        for img in images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url if url.startswith("http") else get_base64(url)}}
                        for url in images
                    ],
                }
            )
            messages[-1]['content'].append({"type":"text","text":instrution})
    else:
        messages.append({"role": "user", "content": instrution})
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=512,
    )
    return chat_response.choices[0].message.content

def get_judge(image_path, question, answer, response):
    prompt = f"## Question\n{question}\n## AI Response\n{response}\n## Reference Answer\n{answer}"
    #score = generate(prompt, image_path)
    score = generate_response(prompt, system_prompt=system_prompt)
    return score

def process_data(image_path, question, answer, response):
    score = get_judge(image_path, question, answer, response)
    raw_score = parse_score(score)
    parsed_score = parse_score(raw_score)
    return {"raw_score":raw_score, "parsed_score":parsed_score}
def superclue_gpt4o_judge(test_file, eval_file):
    df = pd.read_excel(eval_file)
    image_root = test_file.replace("LMUData","LMUData/images").replace(".tsv","")
    df['image_path'] = df['index'].apply(lambda x:os.path.join(image_root, f"{x}.jpg"))
    df['response'] = df['prediction']
    results = parallel_apply(df[["image_path","question","answer","response"]], process_data, num_threads=4)
    return results

def SuperCLUE_eval(image_path, line):
    return get_judge(image_path, line['question'], line['answer'], line['prediction'])

def func(question, answer, prediction):
    score =  get_judge("", question, answer, prediction)
    score = parse_score(score)
    return score
def judge_all(df):
    #这里我们根据df中的三个字段：question, answer, prediction，来进行预测
    df['result'] = parallel_apply(df[['question','answer','prediction']], func, 100)
    df['score'] = df['result'].apply(parse_score)
    return df