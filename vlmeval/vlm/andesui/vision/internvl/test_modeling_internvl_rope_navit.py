import os
os.chdir("/mnt/data/group/wangnan/code/VLMS/AndesVL-V1/model/vision/internvl")
from configuration_intern_vl import InternVisionConfig
from modeling_intern_vl_rope_navit import InternVisionModel
#config = InternVisionConfig.from_pretrained("/mnt/data/group/wangnan/model/InternVL2-40B-AndesVL-Init/InternVit-6B-980px-V1.5-Navit-Leaf-Fix")
#config = InternVisionConfig.from_pretrained("/mnt/data/group/wangnan/model/InternVL2-8B-AndesVL-Init/InternVit-300M-980px-V2-Navit-Leaf")
#visual = InternVisionModel._from_config(config)
visual = InternVisionModel.from_pretrained("/mnt/data/group/wangnan/model/InternVL2.5/4B/InternViT-300M-NaViT-RoPE",device_map="auto",attn_implementation="flash_attention_2")
#print(visual)