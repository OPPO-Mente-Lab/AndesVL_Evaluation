from modeling_qwen2_5_vl_navit_rope import Qwen2_5_VLVisionModel
#from configuration_qwen2_vl import Qwen2VLVisionConfig
#config = Qwen2VLVisionConfig()
#visual = Qwen2VLVisionModel._from_config(config)
visual = Qwen2_5_VLVisionModel.from_pretrained("/mnt/data/group/wangnan/model/Qwen2.5-VL-Separate/7B/Qwen2.5-VL-ViT",device_map={"":"cuda:0"},attn_implementation="flash_attention_2")