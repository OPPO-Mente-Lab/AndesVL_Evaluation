from vllm import ModelRegistry
from .andesvl_internvit import AndesVLForConditionalGeneration
ModelRegistry.register_model("AndesVLForConditionalGeneration",AndesVLForConditionalGeneration)
import runpy
if __name__=="__main__":
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')