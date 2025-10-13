import os
import re
import json
from PIL import Image
from .base import BaseModel
from vllm import LLM, SamplingParams, ModelRegistry
import torch
import sys
from ..smp import *
from ..dataset import DATASET_TYPE

if os.environ.get("model_type") == "andesvl-qwen2vlvit":
    sys.path.append("/mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/vllm/andesvl_vllm")
    from .andesvl_vllm_v090.andesvl_qwen2vlvit import AndesVLForConditionalGeneration as AndesVLForConditionalGeneration
    ModelRegistry.register_model("AndesVLForConditionalGeneration", AndesVLForConditionalGeneration)
elif os.environ.get("model_type") == "andesvl-internvit":
    sys.path.append("/mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/vllm/andesvl_vllm")
    from .andesvl_vllm_v090.andesvl_internvit import AndesVLForConditionalGeneration as AndesVLForConditionalGeneration
    ModelRegistry.register_model("AndesVLForConditionalGeneration", AndesVLForConditionalGeneration)
elif os.environ.get("model_type") == "andesvl-internvit-qwen3":
    sys.path.append("/mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/vllm/andesvl_vllm_v090")
    from .andesvl_vllm_v090.andesvl_internvit import AndesVLForConditionalGeneration as AndesVLForConditionalGeneration
    ModelRegistry.register_model("AndesVLForConditionalGeneration", AndesVLForConditionalGeneration)
elif os.environ.get("model_type") == "andesvl-aimv2-qwen3":
    from .andesvl_vllm_v090.andesvl_aimv2 import AndesVLForConditionalGeneration as AndesVLForConditionalGeneration
    ModelRegistry.register_model("AndesVLForConditionalGeneration", AndesVLForConditionalGeneration)
elif os.environ.get("model_type") == "andesvl-siglip2-qwen3":
    from .andesvl_vllm_v090.andesvl_siglip2 import AndesVLForConditionalGeneration as AndesVLForConditionalGeneration
    ModelRegistry.register_model("AndesVLForConditionalGeneration", AndesVLForConditionalGeneration)


def extract_boxed_content(ans: str):
    idx = ans.rfind(r'\boxed{')
    if idx == -1:
        return ans

    idx += len(r'\boxed{')
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


class AndesVL_V1_vllm(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    def __init__(self, model_path, **kwargs):
        model_dir =  os.environ.get("model_dir")
        model_type = os.environ.get("model_type")
        tp_size = int(os.environ.get("tp_size", torch.cuda.device_count()))
        self.llm = LLM(
            model = model_dir,
            trust_remote_code = True,
            tensor_parallel_size = tp_size,
            limit_mm_per_prompt = {"image": 50},
            gpu_memory_utilization=0.8
        )
        
        self.extra_prompt = os.environ.get("extra_prompt","").strip()
        self.extra_start = os.environ.get("extra_start","")
        self.max_size = os.environ.get("max_size", "1792")
        self.min_size = os.environ.get("min_size", "0")
        self.image_scale = os.environ.get("image_scale", "1")
        self.cot_prompt = "\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."

    def add_custom_prompt(self, prompt, dataset=None):
        if self.extra_prompt:
            prompt += f"\n{self.extra_prompt.strip()}"
        
        thinking_mode = False
        cur_extra_start = ""
        thinking_datasets = ['MMMU_DEV_VAL', 'MMMU_Pro_10c', 'MathVista_MINI', 'MathVision', 'WeMath', 'MathVerse_MINI', 'DynaMath', 
                             'LogicVista', 'MMBench', 'MME', 'ChartQA', 'InfoVQA', 'MMStar', 'MUIRBench', 'HallusionBench']
        if dataset is not None and listinstr(['ScreenSpot', 'AndesUI_test_grounding', 'AndesUI_test_QA'], dataset):
            cur_extra_start = '<|box_start|>('
        elif '<think>' in self.extra_start:
            if listinstr(thinking_datasets, dataset):
                cur_extra_start = self.extra_start
                thinking_mode = True
        else:
            cur_extra_start = self.extra_start

        if listinstr(['ChartQA', 'WeMath'], dataset):
            if thinking_mode:
                prompt = prompt.replace('Answer the question using a single word or phrase.', '')
                prompt = prompt.strip('\n') + self.cot_prompt
        elif listinstr(['MMStar', 'MMBench', 'AI2D_TEST', 'BLINK', 'MMMU_DEV_VAL', 'SEEDBench', 'CCBench'], dataset):
            if listinstr(['MMStar'], dataset) and not thinking_mode:
                prompt = prompt.replace('Please select the correct answer from the options above.',
                                        "Please read the multiple-choice question carefully, reason step by step, and conclude by stating 'The answer is' followed by the correct letter only.")  
            elif 'Options:' in prompt or "Choices:" in prompt:
                prompt = prompt.replace('Please select the correct answer from the options above.', "")
                prompt = prompt.strip('\n') + '\nAnswer with the option’s letter from the given choices directly.'
            else:
                prompt = prompt.strip('\n') + '\nAnswer the question using a single word or phrase directly.'
        elif listinstr(['HallusionBench', 'POPE'], dataset):
            if thinking_mode:
                prompt += "\nConclue with 'The answer is Yes' or 'The answer is No'."
            else:
                prompt += '\nConclude with "The answer is yes" or "The answer is no", and give a brief explanation.'
        elif listinstr(['OCRBench'], dataset):
            if 'write out' in prompt:
                pass
            elif 'Answer this question using the text in the image directly.' not in prompt:
                prompt += '\nAnswer this question based on the text in the image directly.'

        if os.environ.get("completion", 'False') != "True":   
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + cur_extra_start
        return prompt

    def message_to_promptimg(self, message, dataset):
        s = []
        images = []
        for m in message:
            if m['type']=="image":
                s.append("<image>")
                images.append(Image.open(m['value']))
            elif m['type']=='text':
                s.append(m['value'])
            else:
                raise Exception(m)
        d =  {"prompt":"\n".join(s).strip()}
        if images:
            d['multi_modal_data'] = {"image": images}
        d['prompt'] = self.add_custom_prompt(d['prompt'], dataset)
        return d

    def set_infer_params(self, dataset):
        # set inference params
        math_datasets = ['MMMU_DEV_VAL', 'MMMU_Pro_10c', 'MathVista_MINI', 'MathVision_MINI', 'MathVision', 'MathVerse_MINI', 'WeMath',
                         'DynaMath', 'LogicVista']
        if listinstr(math_datasets, dataset):
            default_temperature = 0.6
            default_top_p = 0.95
            default_top_k = 20
            default_presence_penalty = 1.0
        else:
            default_temperature = 0
            default_top_p = 1.0
            default_top_k = -1
            default_presence_penalty = 0

        sample_params_dict = dict(temperature=float(os.environ.get('temperature', default_temperature)),
                                  max_tokens=int(os.environ.get("max_new_tokens", 2048)),
                                  top_p=float(os.environ.get('top_p', default_top_p)),
                                  top_k=int(os.environ.get('top_k', default_top_k)),
                                  presence_penalty=float(os.environ.get('presence_penalty', default_presence_penalty)),
                                  n=1)
        cur_sampling_params = SamplingParams(**sample_params_dict)

        # set other params
        if dataset in ['DynaMath']:
            os.environ['max_size'] = '3000'
        else:
            os.environ['max_size'] = str(self.max_size)
        if DATASET_TYPE(dataset)=='GUI' or listinstr(['ScreenSpot'], dataset):
            os.environ['min_size'] = '1792'
        else:
            os.environ['min_size'] = str(self.min_size)
        
        # set image expand scale
        img_expand_2 = ['InfoVQA', 'OCRBench', 'AI2D_TEST', 'SEEDBench2_Plus', 'CCBench', 'RealWorldQA', 'MMVet', 'MME', 'ChartQA']
        if listinstr(img_expand_2, dataset):
            os.environ['image_scale'] = '2'
        else:
            os.environ['image_scale'] = str(self.image_scale)
        return cur_sampling_params

    def batch_generate(self, messages, dataset=None):
        # 处理输入
        cur_sampling_params = self.set_infer_params(dataset)
        batch_data = [self.message_to_promptimg(message, dataset) for message in messages]
        print(f"dataset: {dataset},  sample params: {cur_sampling_params}")
        responses = self.llm.generate(
            batch_data,
            cur_sampling_params,
            use_tqdm=True
        )
        responses = [r.outputs[0].text for r in responses]
        if dataset is not None and listinstr(['ScreenSpot', 'AndesUI_test_grounding', 'AndesUI_test_QA'], dataset):
            responses = ["(" + r for r in responses]
        for i, response in enumerate(responses[:10]):
            prompt = batch_data[i]['prompt']
            if 'multi_modal_data' in batch_data[i]:
                images = batch_data[i]['multi_modal_data']['image']
                # 把prompt中的<image>替换为<img>image</img>
                for img in [m['value'] for m in messages[i] if m['type']=="image"]:
                    prompt = prompt.replace("<image>", f"<img>{img}</img>", 1)
            print(f"Prompt:\n{prompt.strip()}\nResponse:\n{response}\nExtract:\n{extract_boxed_content(response)}")
        if '</think>' in responses[0]:
            responses = [output[re.search(r'</think>', output).end():] if re.search(r'</think>', output) else output for output in responses]
        # if "<answer>" in responses[0]:
        responses = [re.findall('<answer>(.*?)</answer>', output)[0] if re.findall('<answer>(.*?)</answer>', output) else output for output in responses]
        responses = [re.findall('The answer is (.{1,100})', output, re.M)[0] if re.findall('The answer is (.{1,100})', output, re.M) else output for output in responses]
        responses = [re.findall('the answer is (.{1,100})', output, re.M)[0] if re.findall('the answer is (.{1,100})', output, re.M) else output for output in responses]
        responses = [output.rstrip('.') if output.strip().endswith('.')  else output for output in responses]
        responses = [extract_boxed_content(r) for r in responses]
        return responses
