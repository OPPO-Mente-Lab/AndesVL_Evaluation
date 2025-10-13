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
import json
import sys
import os
from PIL import Image
from io import BytesIO
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import pandas as pd
from io import BytesIO
from PIL import Image
from openai import OpenAI
import sys
sys.path.append("/mnt/data/group/wangnan/code/tools")
from thread_parallel_apply import parallel_apply

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
                        "role": "user",
                        "content": instruction,
                        "images": [
                            {
                                "url": get_base64(img_path),
                                "detail": "auto", #取值为：low/high/auto
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

def get_judge(image_path, user_query, label, output):
    pre_prompt = """我现在在训练一个多模态大语言模型，我需要你根据图片内容给我的模型的输出进行打分，给你的输入是json格式：{"query":"用户
        输入的问题","label":"标注的答案","output":"待评价的模型输出答案"}。你需要根据图片和这些信息以及你自己的知识来对我训练的模型的输出
        进行打分并给出理由，满分5分，如果没有图片就只根据问题和答案进行打分，打分标准如下：
        评分标准（满分5分） 
        准确性（Accuracy, 2分） 
        2分：模型的输出完全正确，与标注好的答案一致，没有明显错误。
        1分：输出基本正确，但有小的细节错误或遗漏，不影响整体理解。 
        0分：模型的输出完全错误，没有正确回答问题，或者回答的内容与问题无关。 
        相关性（Relevance, 2分） 
        2分：模型的回答直接相关于用户的问题，提供了相关的信息。 
        1分：模型的回答部分相关，有一些多余的或不相关的信息，但是主要部分还是有用的。
        0分：模型的回答完全不相关，提供的信息对解决问题没有帮助。 
        语言质量（Language Quality, 1分） 
        1分：模型的输出在字段格式上与标注好的答案一致，并且内容表达清晰没有语病。
        0.5分：模型的输出在字段格式上与标注好的答案略有不通，或者语法上有一些小错误。
        0分：模型的输出在字段格式上与标注答案有明显的差异，或者语法不规范有较多错误。
        打分结果必须严格以字符串的格式输出。
        示例：输入：{"query":"这只猫在做什么？","label":"这只猫在阳光下玩耍。","output":"这只猫在阳光下跳跃。"} ,
        你的输出：{"准确性（1分）":"1","准确性理由":"输出描述了猫咪在阳光下的行为，但使用了“跳跃”而非“玩耍”，虽有差异但总体意思接近。",
        "相关性（2分）":"2","相关性理由":"输出内容与问题和图片高度相关。",
        "语言质量（1分）":"1","语言质量理由":" 输出格式上基本符合标准答案且内容表达清晰没有语病。", "总分": "4.0"}。现在输入："""
    prompt = pre_prompt + str({"query":user_query,"label":label,"output":output})
    #score = generate(prompt, image_path)
    score = generate_response(prompt, images=[])
    return score
   
def CUI_eval(image_path, line):
    try:
        return get_judge(image_path, line['question'], line['answer'], line['prediction'])
    except Exception as e:
        print(e)
        return str(e)


def func(question, answer, prediction):
    score =  get_judge("", question, answer, prediction)
    score = parse_score(score)
    return score

def judge_all(df):
    #这里我们根据df中的三个字段：question, answer, prediction，来进行预测
    df['result'] = parallel_apply(df[['question','answer','prediction']], func, 100)
    df['score'] = df['result'].apply(parse_score)
    return df