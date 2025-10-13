import torch
from transformers import AutoModelForCausalLM
from .base import BaseModel

class megrez_3b(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self,model_path='/mnt/data/group/models/Megrez-3B-Omni/'):
        self.model_path= model_path
        self.MAX_NEW_TOKENS = 100
        self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).eval().cuda()

        
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        # Chat with text and image
        messages = [
            {
                "role": "user",
                "content": {
                    "text": prompt,
                    "image": image_path,
                },
            },
        ]
        response = ''
        try:
            response = self.model.chat(
                messages,
                sampling=False,
                max_new_tokens=self.MAX_NEW_TOKENS,
                temperature=0,
            )
            response = response.replace('<|turn_end|>','')
        except Exception as e:
            print(e)
        
        return response
