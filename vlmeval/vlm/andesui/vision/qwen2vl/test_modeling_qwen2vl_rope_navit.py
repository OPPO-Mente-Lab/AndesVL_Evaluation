from modeling_qwen2_vl_rope_navit import Qwen2VLVisionModel
from configuration_qwen2_vl import Qwen2VLVisionConfig
#config = Qwen2VLVisionConfig()
#visual = Qwen2VLVisionModel._from_config(config)
visual = Qwen2VLVisionModel.from_pretrained("/mnt/data/group/wangnan/model/Qwen2-VL-Separate/7B/vision",device_map={"":"cuda:0"},attn_implementation="flash_attention_2")