from transformers import AutoModel
from .andesvl_v1 import *

class AndesVL_V1_hf(AndesVL_V1):
    def __init__(self, model_path, **kwargs):
        self.max_size = int(os.environ.get("max_size",733))
        self.min_pixels = None
        device = torch.cuda.current_device()
        self.device = device
        model_dir =  os.environ.get("model_dir")
        
        ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": int(os.environ.get("zero_stage", 0)),
            },
            "train_micro_batch_size_per_gpu": 1,
        }

        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = CLIPImageProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
        assert "<|im_end|>" in self.tokenizer.get_vocab()
        if self.tokenizer.eos_token!="<|im_end|>": 
            self.tokenizer.eos_token = "<|im_end|>"
        self.patch_size = 14 if not hasattr(self.processor, 'patch_size') else self.processor.patch_size
        self.model.eval()
        if dist.is_initialized():
            ds_engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
            ds_engine.module.eval()
            self.model  = ds_engine.module
        else:
            self.model.to(self.device)
        temperature=float(os.environ.get('temperature', 0.0))
        do_sample= eval(os.environ.get('do_sample', str(temperature > 0)))
        kwargs_default = dict(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=int(os.environ.get("max_new_tokens", 2048)),
            top_p=float(os.environ.get('top_p', 1.0)),
            repetition_penalty=float(os.environ.get('repetition_penalty', 1.0))
        )
        kwargs_default.update(kwargs)
        self.extra_prompt = os.environ.get("extra_prompt","").strip()
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
