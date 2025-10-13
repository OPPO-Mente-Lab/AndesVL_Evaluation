from functools import partial
import json
from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class VideoSummaryBench(TextBaseDataset):
    TYPE = 'Summary'
    DATASET_URL = {
        'VideoSummaryBench': "/mnt/data/group/vlm_data/business_scenario_data/text_nlora_source_data/summary_test/test-video_summary.tsv"
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        final_score_dict = {} # 暂时没有自动测评方式，这个只能先输出结果人工评测
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class RecordSummaryBench(TextBaseDataset):
    TYPE = 'Summary'
    DATASET_URL = {
        'RecordSummaryBench': "/mnt/data/group/vlm_data/business_scenario_data/text_nlora_source_data/summary_test/test-record_summary.tsv"
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        final_score_dict = {} # 暂时没有自动测评方式，这个只能先输出结果人工评测
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class ThirdRecordSummaryBench(TextBaseDataset):
    TYPE = 'Summary'
    DATASET_URL = {
        'ThirdRecordSummaryBench': "/mnt/data/group/vlm_data/business_scenario_data/text_nlora_source_data/summary_test/test-3rd_record_summary.tsv"
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        final_score_dict = {} # 暂时没有自动测评方式，这个只能先输出结果人工评测
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class EntityCombinationSummaryBench(TextBaseDataset):
    TYPE = 'Summary'
    DATASET_URL = {
        'EntityCombinationSummaryBench': "/mnt/data/group/vlm_data/business_scenario_data/text_nlora_source_data/summary_test/test-entity_combination.tsv"
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        final_score_dict = {} # 暂时没有自动测评方式，这个只能先输出结果人工评测
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict