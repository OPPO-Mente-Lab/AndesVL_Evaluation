from functools import partial
import json
from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich
import sacrebleu
import ast


class TranslationBench(TextBaseDataset):
    TYPE = 'Translation'
    DATASET_URL = {
        'TranslationBench': "/mnt/data/group/vlm_data/business_scenario_data/translation_data/testset/supplementary_test/l1_language_delete_language_label_addline.tsv",
        'TranslationBench_C': "/mnt/data/group/vlm_data/business_scenario_data/translation_data/testset/supplementary_test/l1_language_with_language_label_no_instruction.tsv" 
    }
    TRANSLATION_LANGUAGES = ['en-zh', 'zh-en', 'en-ko', 'ko-en', 'en-id', 'id-en', 'en-hi', 'hi-en', 'en-es', 'es-en', 'en-pt', 'pt-en', 'en-th', 'th-en', \
        'en-ms', 'ms-en', 'en-it', 'it-en', 'en-fr', 'fr-en', 'en-vi', 'vi-en','en-ru', 'ru-en', 'zh-de', 'de-zh', 'zh-ja', 'ja-zh', 'zh-ko', 'ko-zh', \
            'zh-id', 'id-zh', 'zh-hi', 'hi-zh', 'zh-es', 'es-zh', 'zh-pt', 'pt-zh', 'zh-th', 'th-zh', 'zh-ms', 'ms-zh', 'zh-it', 'it-zh', 'zh-fr', 'fr-zh', \
                'zh-vi', 'vi-zh', 'zh-ru', 'ru-zh']
                
                
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        # msgs[-1]['value'] += '\n直接输出答案，不要输出其他'
        # msgs[-1]['value'] += '\n直接输出答案，不要输出其他'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        # 使用yunbei的评测脚本进行评测
        os.system(f'bash scripts/cal_nmt_bleu.sh {eval_file} {os.getcwd()}')
        
        TranslationBench_score = {}
        TranslationBench_count = {}
        for translation_language in self.TRANSLATION_LANGUAGES:
            TranslationBench_score['bleu_' + translation_language] = 0.0
            TranslationBench_count[translation_language] = 0 
        TranslationBench_score['all'] = 0.0
        TranslationBench_count['all'] = 0

        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = [str(line['prediction'])]
            predict = [str(line['prediction'])]
            answers = [ast.literal_eval(line['answer'])] # 标准答案可能是多个，这里用literal_eval转成list，外部再接一个list
            translation_language = line['type']
            tokenize = translation_language.split('-')[-1] # 翻译成的语种
            if tokenize == 'zh':
                bleu_score = sacrebleu.corpus_bleu(predict, answers, tokenize=tokenize)
            elif tokenize == 'ko':
                bleu_score = sacrebleu.corpus_bleu(predict, answers, tokenize="ko-mecab")
            elif tokenize == 'ja':
                bleu_score = sacrebleu.corpus_bleu(predict, answers, tokenize="ja-mecab")
            else:
                bleu_score = sacrebleu.corpus_bleu(predict, answers)
            TranslationBench_score['bleu_' + translation_language] += bleu_score.score
            TranslationBench_score['all'] += bleu_score.score
            TranslationBench_count[translation_language] += 1
            TranslationBench_count['all'] += 1

                
        final_score_dict = {}
        for translation_language in self.TRANSLATION_LANGUAGES:
            if TranslationBench_count[translation_language] > 0:
                final_score_dict['bleu_' + translation_language] = TranslationBench_score['bleu_' + translation_language] / TranslationBench_count[translation_language]
            else:
                final_score_dict['bleu_' + translation_language] = 0
        final_score_dict['all'] = TranslationBench_score['all'] / TranslationBench_count['all']
        
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict