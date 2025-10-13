# The Evaluation Toolkit for AndesVL Series Models

# 快速开始

## 第0步 设置必要的密钥

**设置密钥**: 要使用 API 模型（如 GPT-4v, Gemini-Pro-V 等）进行推理，或使用 LLM API 作为评判者或选择提取器，你需要首先设置 API 密钥。你可以将所需的密钥放在 `$VLMEvalKit/.env` 中，或直接将它们设置为环境变量。

## 第1步 配置环境

**配置环境**: `pip install -r requirements.txt`

## 第2步 评测

#### 参考 `scripts` 目录下测试脚本
- `scripts/run_andesvl_thinking.sh`: 使用vLLM对thinking模型进行评测的脚本
- `scripts/run_andesvl_instruct_vllm.sh`: 使用vLLM对instruct模型进行评测的脚本
- `scripts/run_andesvl_instruct_ds.sh`: 使用deepspeed对instruct模型进行评测的脚本

#### 性能差距
在不同的运行环境中，模型的性能表现可能会有所差异。这种差异可能与`transformers`, `cuda`, `torch`等版本的变化有关。
建议优先查看运行完成后的本地生成记录`{model}_{dataset}.xlsx`或者评估记录`{model}_{dataset}_{judge_model}.xlsx`，可以更好地理解评估结果并发现问题。


## Acknowledgement

The project build upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), which you can visit for relevant details. 
