<div align="center">
  <h1>The Evaluation Toolkit for AndesVL Series Models</h1>
<a href='https://arxiv.org/abs/2510.11496'><img src='https://img.shields.io/badge/arXiv-2510.11496-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/OPPOer'><img src='https://img.shields.io/badge/🤗%20HuggingFace-AndesVL-ffd21f.svg'></a>
</div>

# 🎬 Quick Start

## Step 0: Set Up Necessary Keys

**Set Keys**: To use API models (such as GPT-4v, Gemini-Pro-V, etc.) for inference, or to use an LLM API as a judge or selector extractor, you first need to set up your API keys. You can place the required keys in `$VLMEvalKit/.env` or set them directly as environment variables.

## Step 1: Configure Environment

**Configure Environment**: `pip install -r requirements.txt`

## Step 2: Evaluation

#### Refer to Test Scripts in the `scripts` Directory
- `scripts/run_andesvl_thinking.sh`: Script for evaluating the thinking model using vLLM.
- `scripts/run_andesvl_instruct_vllm.sh`: Script for evaluating the instruct model using vLLM.
- `scripts/run_andesvl_instruct_ds.sh`: Script for evaluating the instruct model using Deepspeed.

#### Performance Discrepancies
Model performance may vary across different execution environments. Such discrepancies might be related to changes in versions of `transformers`, `cuda`, `torch`, etc.
It is recommended to prioritize checking the locally generated records `_.xlsx` or evaluation records `__.xlsx` upon completion, which can provide a better understanding of the evaluation results and help identify issues.

## Acknowledgement

The project build upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), which you can visit for relevant details.