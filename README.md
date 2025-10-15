<div align="center">
  <h1>The Evaluation Toolkit for AndesVL Series Models</h1>
<a href='https://arxiv.org/abs/2510.11496'><img src='https://img.shields.io/badge/arXiv-2510.11496-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/OPPOer'><img src='https://img.shields.io/badge/🤗%20HuggingFace-AndesVL-ffd21f.svg'></a>
</div>

>**AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model**  
> AndesVL Team, OPPO AI Center

We are very excited to introduce **AndesVL**, a state-of-the-art model designed for mobile-side applications with 0.6B to 4B parameters.

## 🔥 News
- 2025/10/13: **AndesVL Technical Report** is now available at [arxiv](https://arxiv.org/abs/2509.14033) and **AndesVL Models** is  available at [huggingface](https://huggingface.co/collections/OPPOer/andesvl-68ecb641e4b854cb0c7c2e7d).

## 🎬 Quick Start
Inference using the non-thinking version of AndesVL

```python
# require transformers>=4.52.4

import torch
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

model_dir = "OPPOer/AndesVL-4B-Instruct"

model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
image_processor = CLIPImageProcessor.from_pretrained(model_dir, trust_remote_code=True)

messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述这张图片。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://i-blog.csdnimg.cn/blog_migrate/2f4c88e71f7eabe46d062d2f1ec77d10.jpeg" # image/to/path
                            },
                        }
                    ],
                },
        ]
response = model.chat(messages, tokenizer, image_processor, max_new_tokens=1024, do_sample=True, temperature=0.6)
print(response)
```

Inference using the thinking version of AndesVL
```python
# require transformers>=4.52.4
import torch
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

model_dir = "OPPOer/AndesVL-4B-Thinking"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
image_processor = CLIPImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述这张图片。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://i-blog.csdnimg.cn/blog_migrate/2f4c88e71f7eabe46d062d2f1ec77d10.jpeg" # image/to/path
                            },
                        }
                    ],
                },
        ]
response = model.chat(messages, tokenizer, image_processor, max_new_tokens=1024, do_sample=True, temperature=0.6, Thinking=True)
print(response)
```
## 📚Evaluation

#### Setup essential keys
To infer with API models (GPT-4v, Gemini-Pro-V, etc.) or use LLM APIs as the judge or choice extractor, you need to first setup API keys.You can place the required keys in `$VLMEvalKit/.env` or directly set them as the environment variable.

#### Environment
Prepare the environment, install the required libraries:

```shell
$ cd AndesVL_Evaluation
$ conda create --name AndesVL_Evaluation python==3.11
$ conda activate AndesVL_Evaluation
$ pip install -r requirements.txt
```

#### Run evaluation
Evaluation script for thinking models using vLLM

```shell
bash scripts/run_andesvl_thinking.sh
```

Evaluation script for instruct models using vLLM

```shell
bash scripts/run_andesvl_instruct_vllm.sh
```

Evaluation script for instruct models using deepspeed

```shell
bash scripts/run_andesvl_general.sh
```

#### Performance Discrepancies

Model performance may vary across different environments. These differences could be attributed to variations in versions of libraries such as `transformers`, `cuda`, and `torch`. 

If you encounter unexpected performance, we recommend first reviewing the local generation records (`{model}_{dataset}.xlsx`) or the evaluation records (`{model}_{dataset}_{judge_model}.xlsx`). This may help you better understand the evaluation outcomes and identify potential issues.

## Acknowledgement

The project build upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), which you can visit for relevant details.

## Citation
If you find our work helpful, feel free to give us a cite.

```
@misc{jin2025andesvltechnicalreportefficient,
      title={AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model}, 
      author={AndesVL Team, OPPO AI Center},
      year={2025},
      eprint={2510.11496},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.11496}, 
}
```