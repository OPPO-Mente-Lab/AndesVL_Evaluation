from modeling_siglip2_navit_rope import Siglip2VisionModel
from configuration_siglip2_navit_rope import Siglip2VisionConfig
#config = Qwen2VLVisionConfig()
#visual = Qwen2VLVisionModel._from_config(config)
#visual = Qwen2VLVisionModel.from_pretrained("/mnt/data/group/wangnan/model/Qwen2-VL-Separate/7B/vision",device_map={"":"cuda:0"},attn_implementation="flash_attention_2")
#config = Siglip2VisionConfig()
#visual = Siglip2VisionModel._from_config(config)
visual = Siglip2VisionModel.from_pretrained("/mnt/data/group/wangnan/model/siglip2-base-patch16-navit-rope")
pass