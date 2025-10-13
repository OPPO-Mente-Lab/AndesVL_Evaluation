# instruct模型vllm测试脚本，适用于pt，sft和mpo instruct模型
export VLMEValKit_ROOT=/mnt/data/group/chenglong/evaluation/andesvl_git  # 测试代码目录，更换为自己本地的路径
cd $VLMEValKit_ROOT    # 切换到根目录

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export LMUData=/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData
export hd=False
export VLLM_USE_V1=0
# export tp_size=4             # set only for 0.6B model

# 设置测试参数
export max_size=1792
export batch_size=-1
export max_new_tokens=16384    # 对于不需要推理测试集可以设置2048
# greedy for instruct model
export temperature=0
export top_p=1.0
export top_k=-1
export presence_penalty=0

# 模型路径：4b instruct mpo model
export model_dir=/mnt/data/group/lichao/code/AndesVL-V1-0514/mpo_checkpoints_1001/4b/202509/20250928171951-dlc6bd2bthum8zv0/update-2000-loss-7.5538-tokens-7.34E+08/andesvl-aimv2-qwen3 # 4b instruct mpo model
export pred_root=$VLMEValKit_ROOT/results/andesvl-4b-mpo/base_vllm  # 结果保存路径
export model_type=andesvl-aimv2-qwen3  # 模型类型, andesvl-aimv2-qwen3 for 1-4B, andesvl-siglip2-qwen3 for 0.6B

# 保存当前测试脚本
cur_file=$(realpath "$0")
mkdir -p $pred_root
cp $cur_file $pred_root

# mode: 支持infer(仅推理), eval(仅评估), all(推理+评估)
# data: 测试集名称，多个测试集用空格隔开
python run.py --model andesvl_v1_vllm \
              --mode all \
              --data MMMU_DEV_VAL \
              2>&1 | tee $pred_root/log.txt

# Test Dataset
# Math:          MMMU_DEV_VAL MathVista_MINI MMMU_Pro_10c WeMath LogicVista MathVerse_MINI_Vision_Only DynaMath MathVision MathVision_MINI MathVerse_MINI
# TextRich:      AI2D_TEST AI2D_TEST_NO_MASK  ChartQA_TEST TextVQA_VAL InfoVQA_VAL InfoVQA_TEST OCRBench SEEDBench2_Plus DocVQA_VAL DocVQA_TEST
# MultiImage:    BLINK MMT-Bench_VAL MUIRBench Q-Bench1_VAL MMT-Bench_VAL_MI 
# RealWorld:     RealWorldQA R-Bench-Dis
# Understanding: MME MMBench_DEV_CN_V11 MMBench_TEST_CN_V11 MMBench_TEST_EN_V11 MMVet MMStar CCBench 
# Hallusion:     HallusionBench POPE CRPE_RELATION
# Multilingual:  MMMB_en MMMB_cn MMMB_ar MMMB_pt MMMB_ru MMMB_tr MMBench_dev_en MMBench_dev_cn MMBench_dev_ar MMBench_dev_pt MMBench_dev_ru MMBench_dev_tr MTVQA_TEST
# GUI:           ScreenSpot ScreenSpot_v2 ScreenSpot_Pro 
# AndesUI:       AndesUI_test_grounding_data AndesUI_test_QA_data AndesUI_test_referring_data