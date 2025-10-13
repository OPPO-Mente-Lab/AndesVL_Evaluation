import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"]="0" #v1下的代码，相同输入情况下，会出现异常，暂时不支持。
import PIL
from vllm import ModelRegistry
from .andesvl_siglip2 import AndesVLForConditionalGeneration
ModelRegistry.register_model(
    "AndesVLForConditionalGeneration", AndesVLForConditionalGeneration
)

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=1, max_tokens=512)
    llm = LLM(
        model="/mnt/data/group/wangnan/code/VLMS/AndesVL-V1/infer/hf/full/andesvl-siglip2-qwen3-temp",
        enforce_eager=True,  
        trust_remote_code=True,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": 10},
    )
    """
    prompts = ["Hello, my name is"]
    outputs = llm.generate(prompts, sampling_params)
    """
    """纯文本的测试
    prompt = "<|im_start|>user\n你好啊，请简单做一个自我介绍。<|im_end|>\n<|im_start|>assistant\n"
    outputs = llm.generate(
        {
            "prompt": prompt,
        },
        sampling_params=sampling_params,
        use_tqdm=True
    )
    """
    """单图的测试
    #prompt = "<|im_start|>user\n<image>\n请详细描述一下这张图片。<|im_end|>\n<|im_start|>assistant\n"
    #image = PIL.Image.open("/mnt/data/group/wangnan/code/VLMS/cat.jpg") 
    prompt = "<|im_start|>user\n<image>\n请识别出图片中的文字，按照从左到右，从上到下的顺序，回复所有识别到的文字。<|im_end|>\n<|im_start|>assistant\n"
    image = PIL.Image.open("/mnt/data/group/wangnan/text.jpg")
    #prompt = "<|im_start|>user\n<image>图中描绘的是什么景象？<|im_end|>\n<|im_start|>assistant\n"
    #image = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )
    """
    """多图的测试 （输出结果很奇怪,和huggingface的输出结果差异较大，但是程序是可以正常运行的，以及分别描述第一或二张图片可以看出两张图片都是存在一定理解的）
    #prompt = "<|im_start|>user\n<image><image>这两张图片的区别是什么？<|im_end|>\n<|im_start|>assistant\n"
    prompt = "<|im_start|>user\n<image><image>第二张图片是什么？<|im_end|>\n<|im_start|>assistant\n"
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [PIL.Image.open("/mnt/data/group/wangnan/code/VLMS/cat.jpg")
                                    ,PIL.Image.open("/mnt/data/group/wangnan/text.jpg")]},
    })
    """
    """batch的单图输入
    prompt = "<|im_start|>user\n<image>\n请详细描述一下这张图片。<|im_end|>\n<|im_start|>assistant\n"
    #image = PIL.Image.open("/mnt/data/group/wangnan/code/VLMS/cat.jpg") 
    prompt = "<|im_start|>user\n<image>\n请识别出图片中的文字，按照从左到右，从上到下的顺序，回复所有识别到的文字。<|im_end|>\n<|im_start|>assistant\n"
    image = PIL.Image.open("/mnt/data/group/wangnan/text.jpg")
    outputs = llm.generate(
        [{
            "prompt": prompt + str(i),
            "multi_modal_data": {"image": image},
        } for i in range(10)],
        sampling_params=sampling_params,
    )
    """
    #"""batch的多图输入 （似乎在处理第二条数据的时候，出现了较大的异常，排查可以发现，此时的input_ids是不正确的，可以初步推断是vLLM在v1 engine下的cache机制导致的bug）
    prompt = "<|im_start|>user\n<image><image>这两张图片的区别是什么？\n<|im_start|>assistant\n"
    #prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>\n<|im_start|>assistant\n"
    outputs = llm.generate(
        [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": [PIL.Image.open("/mnt/data/group/wangnan/code/VLMS/cat.jpg")
                                ,PIL.Image.open("/mnt/data/group/wangnan/text.jpg")]},
        },
        {
            "prompt": prompt,
            "multi_modal_data": {"image": [PIL.Image.open("/mnt/data/group/wangnan/code/VLMS/cat.jpg")
                                ,PIL.Image.open("/mnt/data/group/wangnan/text.jpg")]},
        }
        #{
        #    "prompt": "<image><image><|im_start|>user\n今天天气不错对吧？|im_start|>assistant\n",
        #    "multi_modal_data": {"image": [PIL.Image.open("/mnt/data/group/wangnan/code/VLMS/cat.jpg") 
        #                        ,PIL.Image.open("/mnt/data/group/wangnan/text.jpg")]},
        #},
        ]*2
    )
    #"""
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
