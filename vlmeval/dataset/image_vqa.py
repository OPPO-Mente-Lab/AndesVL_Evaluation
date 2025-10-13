from functools import partial
import json
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class ImageVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
        'GQA_TestDev_Balanced': 'https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv',
    }

    DATASET_MD5 = {
        'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
        'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
        'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
        'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
        'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
        'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
        'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
        'GQA_TestDev_Balanced': 'fead7df22befc1ed3ca2b62ea26fa17b',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        if 'batch_size' in os.environ: #自研模型评测时会使用这个环境变量，做一个后处理
            if listinstr(['ChartQA'], dataset):
                data['prediction'] = [x.rstrip('.') for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        if listinstr(['TextVQA'], dataset):
            res = pool.map(partial(process_line, method='vqa_score'), lines)
        elif listinstr(['ChartQA'], dataset):
            res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
        elif listinstr(['OCRVQA', 'GQA'], dataset):
            res = pool.map(partial(process_line, method='accuracy'), lines)
        elif listinstr(['DocVQA', 'InfoVQA'], dataset):
            res = pool.map(partial(process_line, method='anls'), lines)
        else:  # default using vqa_score to calculate score
            res = pool.map(process_line, lines)
        hit = hit_calculate(res, dataset)
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = hit_calculate(sub, dataset)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, dataset)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret


class OCRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv'
    }
    DATASET_MD5 = {'OCRBench': 'e953d98a987cc6e26ef717b61260b778'}

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        OCRBench_score = {
            'Regular Text Recognition': 0,
            'Irregular Text Recognition': 0,
            'Artistic Text Recognition': 0,
            'Handwriting Recognition': 0,
            'Digit String Recognition': 0,
            'Non-Semantic Text Recognition': 0,
            'Scene Text-centric VQA': 0,
            'Doc-oriented VQA': 0,
            'Key Information Extraction': 0,
            'Handwritten Mathematical Expression Recognition': 0,
        }

        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = eval(line['answer'])
            category = line['category']
            if category == 'Handwritten Mathematical Expression Recognition':
                for j in range(len(answers)):
                    answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
                    predict = predict.strip().replace('\n', ' ').replace(' ', '')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break
            else:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace('\n', ' ')
                    predict = predict.lower().strip().replace('\n', ' ')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break

        final_score_dict = {}
        final_score_dict['Text Recognition'] = \
            (OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
             + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
             + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition'])
        final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
        final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
        final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
        final_score_dict['Handwritten Mathematical Expression Recognition'] = \
            (OCRBench_score['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score'] = \
            (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
             + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
             + final_score_dict['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score Norm'] = (float(final_score_dict['Final Score']) / 10)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class ChineseOCRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'ChineseOCRBench': "/mnt/data/group/wangnan/code/VLMEvalKit/LMUData/ChineseOCRBench.tsv"
    }
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\n直接输出答案，不要输出其他'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        OCRBench_score = {
            'ESTVQA_cn': 0,
            'ReCTS': 0
        }
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = eval(line['answer'])
            category = line['dataset_name']
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace('\n', ' ')
                predict = predict.lower().strip().replace('\n', ' ')
                if answer in predict:
                    OCRBench_score[category] += 1
                    break

        final_score_dict = {}
        ESTVQA_cn_count = sum([1 for line in lines if line['dataset_name'] == 'ESTVQA_cn'])
        ReCTS_count = sum([1 for line in lines if line['dataset_name'] == 'ReCTS'])
        for key, value in OCRBench_score.items():
            final_score_dict[key] = value / (ESTVQA_cn_count if key == 'ESTVQA_cn' else ReCTS_count)
        final_score_dict['all']= sum(OCRBench_score.values())/(ESTVQA_cn_count+ReCTS_count)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict

class SuperCLUE(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'SuperCLUE': "/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/SuperCLUE.tsv"
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        #from .utils.superclue import SuperCLUE_eval, parse_score
        #results = superclue_gpt4o_judge(self.DATASET_URL['SuperCLUE'], eval_file)
        #下面计算score
        #下面保存results和score
        from .utils.superclue import judge_all
        model = "qwen2_5-72b"
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        if not osp.exists(storage):
            data = load(eval_file)
            data = judge_all(data)
            dump(data, storage)
        else:
            data = load(storage)
        # 定义需要的keys
        keys = ['正确性', '相关性', '流畅性', '知识延伸', '输出样式多样化', '多感官信息融合']

        # 确保score列是字典类型，转换失败的行跳过
        def convert_score(x):
            try:
                if isinstance(x, str):
                    return json.loads(x)
                elif isinstance(x, dict):
                    return x
                else:
                    return None
            except:
                return None

        # 应用转换函数
        data['score'] = data['score'].apply(convert_score)

        # 过滤掉score为None的行
        data = data[data['score'].notnull()]
        # 过滤掉不满足条件的行
        data = data[data['score'].apply(lambda x: all(k in x and isinstance(x[k], (int, float)) for k in keys))]
        # 展开score列
        score_df = pd.json_normalize(data['score'])
        # 合并评分维度列到原始数据
        data_expanded = pd.concat([data[['type', 'sub_type']], score_df], axis=1)
        # 按type和sub_type分组，计算评分维度的平均值
        grouped = data_expanded.groupby(['type', 'sub_type']).mean().reset_index()
        # 在grouped中添加每个sub_type的平均分
        grouped['平均分'] = grouped[keys].mean(axis=1)
        # 按type分组，创建嵌套字典
        result_dict = {}
        type_groups = grouped.groupby('type')
        for type_name, group in type_groups:
            type_dict = {}
            sub_type_group = group.groupby('sub_type')
            for sub_type_name, sub_group in sub_type_group:
                scores = sub_group[keys].iloc[0].to_dict()
                type_dict[sub_type_name] = scores
            # 计算该type的平均分
            type_average_score = group[keys].mean().mean()
            type_dict['平均分'] = float(type_average_score)
            result_dict[type_name] = type_dict
        # 计算总体平均分
        overall_average = grouped['平均分'].mean()
        result_dict['总体平均分'] = float(overall_average)
        # 输出结果
        print(json.dumps(result_dict,indent=4,ensure_ascii=False))
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(result_dict, score_pth)
        #return result_dict #返回值打印会出现ascii的情况，暂时但会，上面直接打印就行了。

'''
class CUI(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        '第三轮CUI测试集_2550': "/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/第三轮CUI测试集_2550.tsv",
        'fanpinpai_20240929': "/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/fanpinpai_20240929.tsv",
        "CUI-459": "/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/CUI-459.tsv",
    }
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.cui import judge_all
        model = "qwen2_5-72b"
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        for k in self.DATASET_URL:
            if k in eval_file:
                test_name = k
                break
        if not osp.exists(storage):
            data = load(eval_file)
            data = judge_all(data)
            if test_name=="CUI-459":
                score_list = self.eval_short(data)
                data['score_rule'] = score_list
            dump(data, storage)
        else:
            data = load(storage)
        #print("TODO: 实现分类别的得分输出")
        score = [eval(s) for s in load(storage)['score']]
        keys = ['准确性','相关性','语言质量']
        cum_scores = {k:0 for k in keys}
        cum_nums = {k:0 for k in keys}
        for s in score:
            for k in keys:
                if isinstance(s, dict):
                    mk = [sk for sk in s if k in sk and '理由' not in sk]
                    if len(mk)!=1:
                        print(f"WARN: 跳过{s} {k}")
                        continue
                    cum_scores[k] += float(s[mk[0]])
                    cum_nums[k] += 1
                else:
                    print(f"WARN: 跳过{s} {k}")
                    continue
        final_score = {k:cum_scores[k]/cum_nums[k]for k in keys}
        if test_name=="CUI-459":
            final_score['score_rule'] = data['score_rule'].mean()
        score_pth = storage.replace('.xlsx', '_score.json')
        dump(final_score, score_pth)
        return final_score
    
    @classmethod
    def eval_short(self, data):
        score_list = [0]*len(data)
        for i,r in tqdm(data.iterrows()):
            predict = str(r['prediction'])
            answers = str(r['answer'])
            
            answer = answers.lower().strip()
            predict = predict.lower().strip().replace('\n', ' ')

            # '&':且关系；' '：或关系
            if '&' in answer:
                answer_list = answer.split('&')
                cnt = 0
                for a in answer_list:
                    if a not in predict:
                        break
                    cnt+=1
                if cnt == len(answer_list):
                    score_list[i]=1
            elif ' ' in answer:
                answer_list = answer.split(' ')
                for a in answer_list:
                    if a in predict:
                        if a in predict:
                            score_list[i]=1
                            break
            elif answer in predict:
                score_list[i]=1
        return score_list
'''
class CUI(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        "CUI-459": "/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/CUI-459.tsv"
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        score_list = [0]*len(data)
        for i,r in tqdm(data.iterrows()):
            predict = str(r['prediction'])
            answers = str(r['answer'])
            
            answer = answers.lower().strip()
            predict = predict.lower().strip().replace('\n', ' ')

            # '&':且关系；' '：或关系
            if '&' in answer:
                answer_list = answer.split('&')
                cnt = 0
                for a in answer_list:
                    if a not in predict:
                        break
                    cnt+=1
                if cnt == len(answer_list):
                    score_list[i]=1
            elif ' ' in answer:
                answer_list = answer.split(' ')
                for a in answer_list:
                    if a in predict:
                        if a in predict:
                            score_list[i]=1
                            break
            elif answer in predict:
                score_list[i]=1

        score_dict = {}
        score_dict['score']=sum(score_list)/len(data)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(score_dict, score_pth)

        import pandas as pd
        df2 = data.assign(score=score_list)
        df2.to_excel(score_pth.replace('.json','.xlsx'), index=False)

        return score_dict
    
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\n尽可能简短的回答问题。'
        return msgs

class FunctionCallDataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {"mllm_eval_data_评估明细20241031":"/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/mllm_eval_data_评估明细20241031.tsv"}

    def evaluate(self, eval_file, **judge_kwargs):
        pass

class MathVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVista_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv'
    }
    DATASET_MD5 = {'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathvista import MathVista_auxeval, MathVista_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVista_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MathVista_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class MathVerse(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVerse_MINI': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIV.tsv', # noqa
        'MathVerse_MINI_Vision_Only': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv', # noqa
        'MathVerse_MINI_Vision_Only_cot': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv', # noqa
        'MathVerse_MINI_Vision_Dominant': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVDom.tsv', # noqa
        'MathVerse_MINI_Vision_Intensive': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVInt.tsv', # noqa
        'MathVerse_MINI_Text_Lite': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINITLite.tsv', # noqa
        'MathVerse_MINI_Text_Dominant': 'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINITDom.tsv', # noqa
    }
    DATASET_MD5 = {
        'MathVerse_MINI': '5017caca32b7fa110c350a1bea861b65',
        'MathVerse_MINI_Vision_Only': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Only_cot': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Dominant': 'b8fb63852d261ab2aaefba29cc2414d3',
        'MathVerse_MINI_Vision_Intensive': '01cbd35be202bb0c4873a4186a63bc19',
        'MathVerse_MINI_Text_Lite': '19e4b13bdd30b89a03b2e358bcfefa04',
        'MathVerse_MINI_Text_Dominant': '4f5cd2fa6630ea00bb11d6fde1f6fe6a',
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        if 'cot' in self.dataset_name:
            question = line['query_cot']
        else:
            question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathverse import MathVerse_auxeval_extract, MathVerse_auxeval_score, MathVerse_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.xlsx')
        tmp_file_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.pkl')
        storage_score = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file_score = eval_file.replace(f'.{suffix}', f'_{model}_score.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_extract'] == v['log_extract'] and ans[k]['extract'] == v['extract']

            data['extract'] = [ans[idx]['extract'] for idx in data['index']]
            data['log_extract'] = [ans[idx]['log_extract'] for idx in data['index']]
            dump(data, storage_extract)

        # stage2: score the answer
        if not osp.exists(storage_score):
            data = load(storage_extract)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_score,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k]['score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [ans[idx]['log_score'] for idx in data['index']]
            dump(data, storage_score)

        score = MathVerse_acc(storage_score)
        score_pth = storage_score.replace('.xlsx', '.csv')
        dump(score, score_pth)
        return score


class MathVision(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVision': 'https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv',
        'MathVision_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv'
    }
    DATASET_MD5 = {
        'MathVision': '93f6de14f7916e598aa1b7165589831e',
        'MathVision_MINI': '060fe4fa5d868987ce179307bd5f8a33'
    }

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathv import MATH_V_auxeval, MATH_V_acc

        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MATH-Vision evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MATH_V_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MATH_V_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class LLaVABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'LLaVABench': 'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv'}
    DATASET_MD5 = {'LLaVABench': 'd382a093f749a697820d3dadd61c8428'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.llavabench import (
            build_prompt,
            LLaVABench_atomeval,
            LLaVABench_score,
        )

        suffix = '.' + eval_file.split('.')[-1]
        record_file = eval_file.replace(suffix, '_openai_result' + suffix)
        score_file = eval_file.replace(suffix, '_score.csv')
        nproc = judge_kwargs.pop('nproc', 4)
        system_prompt = 'You are a helpful and precise assistant for checking the quality of the answer.'

        if not osp.exists(record_file):
            data = load(eval_file)
            lines = [data.iloc[i] for i in range(len(data))]
            model = build_judge(temperature=0.2, system_prompt=system_prompt, **judge_kwargs)
            assert model.working(), ('LLaVABench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            prompts = [build_prompt(line) for line in lines]
            tups = [(model, prompt) for prompt in prompts]
            scores = track_progress_rich(LLaVABench_atomeval, tups, nproc=nproc, chunksize=nproc)
            data['gpt4_score'] = [x[0] for x in scores]
            data['score'] = [x[1] for x in scores]
            dump(data, record_file)

        data = load(record_file)
        ret = LLaVABench_score(data).round(1)
        dump(ret, score_file)
        return ret


class MMVet(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMVet': 'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv'
    }
    DATASET_MD5 = {'MMVet': '748aa6d4aa9d4de798306a63718455e3'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmvet import MMVet_auxeval, MMVet_acc

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=3, **judge_kwargs)
            assert model.working(), ('MMVet evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMVet_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMVet_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score


class MTVQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'MTVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MTVQA_TEST.tsv'}
    DATASET_MD5 = {'MTVQA_TEST': 'd87c17dbab934b7cd89c0a3c1c5657f4'}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data and 'category' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        if 'split' in data:
            assert np.all([x.lower() == 'test' for x in data['split']]), 'We only support MTVQA_TEST for now. '
        lt = len(data)
        category_scores = defaultdict(list)
        for i in range(lt):
            line = data.iloc[i]
            ans = line['answer'].strip().lower().replace('.', '')
            pred = line['prediction'].strip().lower().replace('.', '')
            cate = line['category']
            score = 1.0 if ans in pred else 0.0
            category_scores[cate].append(score)
            category_scores['Average'].append(score)
        # Calculate the average score for each category, the score is normalized to [0, 100]
        category_averages = {category: np.mean(scores) * 100 for category, scores in category_scores.items()}

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.json')
        dump(category_averages, result_file)

        return category_averages

    # MT-VQA adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                item['value'] += '\nAnswer the question using a word or phrase in the language of the question.'
        return msgs


class TableVQABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'TableVQABench': 'https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/mentor-vil/datasets/tablevqa-bench.tsv'
    }
    DATASET_MD5 = {'TableVQABench': '2550adc61bdc82d8e62f3b003de7c62d'}

    from .utils.tablevqabench import FINTABNETQA_PROMPT, VTABFACT_PROMPT, VWTQ_PROMPT

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        import pandas as pd
        from .utils.tablevqabench import evaluate_fintabnet, evaluate_tabfact, evaluate_wtq

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        data['prediction'] = data['prediction'].str.replace('^Answer: ', '', regex=True)
        data_group = dict(tuple(data.groupby('split')))
        eval_result = {'split': [], 'average_scores': []}
        for split in ['fintabnetqa', 'vtabfact', 'vwtq', 'vwtq_syn']:
            data_split = data_group[split].to_dict(orient='records')
            if split == 'fintabnetqa':
                split_eval_meta = evaluate_fintabnet(data_split, ['accuracy'])
            elif split == 'vtabfact':
                split_eval_meta = evaluate_tabfact(data_split, ['accuracy'])
            elif split == 'vwtq' or split == 'vwtq_syn':
                split_eval_meta = evaluate_wtq(data_split, ['accuracy'])
            else:
                exit(0)
            eval_result['split'].append(split)
            eval_result['average_scores'].append(split_eval_meta['average_scores'])

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        eval_result = pd.DataFrame(eval_result)
        dump(eval_result, result_file)

        return eval_result

    # TableVQABench adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                if line['split'] == 'fintabnetqa':
                    item['value'] = self.FINTABNETQA_PROMPT.format_map({'question': item['value']})
                elif line['split'] == 'vtabfact':
                    item['value'] = self.VTABFACT_PROMPT.format_map({'question': item['value']})
                elif line['split'] == 'vwtq_syn' or line['split'] == 'vwtq':
                    item['value'] = self.VWTQ_PROMPT.format_map({'question': item['value']})
        return msgs


class CustomVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError


class CRPE(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'CRPE_EXIST': 'https://huggingface.co/datasets/petter12321/crpe_vlmevalkit/resolve/main/CRPE_EXIST.tsv',
        'CRPE_RELATION': 'https://huggingface.co/datasets/petter12321/crpe_vlmevalkit/resolve/main/CRPE_RELATION.tsv'
    }
    DATASET_MD5 = {
        'CRPE_EXIST': '315584e23ac1ff7f8719ed3b7ad90f08',
        'CRPE_RELATION': 'bad7094cde0b572288f4b119c2d0c656'}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.crpe import is_correct
        # find-image, count-text, find-text,
        # infer-choose, count-image, visual-reasoning
        score = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        num = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        final_score_dict = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = str(line['answer'])
            # print("predict =", predict)
            # print("answers =", answers)
            category = line['category']
            if is_correct(answers, predict):
                score[category] += 1
                score['total'] += 1
            num[category] += 1
            num['total'] += 1

        for category in ['exist', 'subject', 'predicate', 'object', 'total']:
            if num[category] != 0:
                final_score_dict[category] = score[category] / num[category]
            else:
                final_score_dict[category] = None

        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict

    def build_prompt(self, line):
        ROOT = LMUDataRoot()
        msgs = super().build_prompt(line)
        for msg in msgs:
            if msg['type'] == 'image':
                msg['value'] = osp.join(osp.join(ROOT, 'images', self.dataset_name), msg['value'])
        return msgs

class OlympiadBench(ImageBaseDataset):
    TYPE = 'VQA_ex_prompt'
    DATASET_URL = {
        'OlympiadBench': 'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench.tsv',
        'OlympiadBench_EN': 'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench_EN.tsv',
        'OlympiadBench_CN': 'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench_CN.tsv'
    }
    DATASET_MD5 = {
        'OlympiadBench': '9735ae0f0299eae1e7d07f5a7feab914',
        'OlympiadBench_EN': '5c68e100d394351fc7049f29d4d4efed',
        'OlympiadBench_CN': 'ea01b16788955702c79650c701e5b623'
    }

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

    def build_prompt(self, line):

        from .utils.olympiadbench import get_answer_type_text, make_input

        self.is_chinese = 'zh' in line['source']
        self.is_math = 'maths' in line['source']
        self.is_theorem_proving = 'TP' in line['source']

        if self.is_chinese:
            subject_content = '数学' if self.is_math else '物理'
            if self.is_theorem_proving:
                prompt = (
                    f"以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。"
                    "证明过程中使用的变量和公式请使用LaTeX格式表示。"
                )
            else:
                answer_type_text = get_answer_type_text(line['answer_type'], is_chinese=True,
                                                        multiple_answer=line['is_multiple_answer'])
                if line['is_multiple_answer']:
                    multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
                else:
                    multiple_answer_text = '\\boxed{答案}'
                unit_text = ''
                if line['unit']:
                    multiple_answer_text += '(单位)'
                    unit_text = '，注意答案的单位不要放在\\boxed{}中'
                prompt = (
                    f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。'
                    f'解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”'
                    f'显式给出结果{unit_text}。'
                )
        else:
            subject_content = 'Math' if self.is_math else 'Physics'
            if self.is_theorem_proving:
                prompt = (
                    f'The following is a theorem proving problem from an International {subject_content} competition. '
                    'Please use logical reasoning and common theorems to prove the proposition in the problem '
                    'according to the given requirements. '
                    'Please use LaTeX format to represent the variables and formulas used in the proof.'
                )
            else:
                if line['is_multiple_answer']:
                    multiple_answer_text = '\\boxed{multiple answers connected with commas}'
                else:
                    multiple_answer_text = '\\boxed{answer}'
                unit_text = ''
                if line['unit']:
                    multiple_answer_text += '(unit)'
                    unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
                answer_type_text = get_answer_type_text(line['answer_type'], is_chinese=False,
                                                        multiple_answer=line['is_multiple_answer'])
                prompt = (
                    f'The following is an open-ended problem from an International {subject_content} competition. '
                    f'{answer_type_text}Please calculate the answer according to the given requirements and '
                    'the information provided. Please use LaTeX format to represent the variables and formulas '
                    'used in the solution process and results. Please end your solution with "So the final answer '
                    f'is {multiple_answer_text}." and give the result explicitly{unit_text}.'
                )

        if self.is_math:
            input = make_input(prompt, line['question'])
        else:
            if 'context' in line.keys() and str(line['context']) != 'nan':  # cannot be null
                input = make_input(prompt, line['context'] + '\n' + line['question'])
            else:
                input = make_input(prompt, line['question'])

        ret = [dict(type='text', value=input)]
        tgt_path = self.dump_image(line)

        ret.extend([dict(type='image', value=s) for s in tgt_path])

        return ret

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.olympiadbench import MathJudger, extract_answer
        judger = MathJudger()

        suffix = eval_file.split('.')[-1]
        name_str1 = 'judge'
        name_str2 = 'score'
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str1}_result.xlsx')
        score_file = eval_file.replace(f'.{suffix}', f'_{name_str2}_result.csv')

        if not osp.exists(result_file):
            data = load(eval_file)
            scorez = []

            for i in tqdm(data.iterrows()):
                line = i[1]
                model_answer = line['prediction']
                is_chinese = 'zh' in line['source']
                model_answer = extract_answer(is_chinese, model_answer, is_deepseek=False)
                answer_type = line['answer_type']

                final_answer = line['final_answer'][2:-2]

                if str(answer_type) != 'nan' and 'Tuple' in answer_type:
                    judge_result = judger.judge(model_answer, final_answer)
                else:
                    if str(line['error']) != 'nan':
                        if ',' in line['error']:
                            precisions = line['error'].split(',')
                            precisions = [float(p) if p else 1e-8 for p in precisions]
                            judge_result = judger.judge(model_answer, final_answer, precisions)
                        else:
                            precision = float(line['error'])
                            judge_result = judger.judge(model_answer, final_answer, precision)
                    else:
                        judge_result = judger.judge(model_answer, final_answer)
                scorez.append(judge_result)

            data['score'] = scorez
            dump(data, result_file)

        judge_file = load(result_file)

        if not osp.exists(score_file):
            name_list = ['OE_MM_maths_en_COMP', 'OE_MM_maths_zh_CEE', 'OE_MM_maths_zh_COMP', 'OE_MM_physics_en_COMP',
                         'OE_MM_physics_zh_CEE','OE_TO_maths_en_COMP', 'OE_TO_maths_zh_CEE', 'OE_TO_maths_zh_COMP',
                         'OE_TO_physics_en_COMP', 'OE_TO_physics_zh_CEE']

            sample_list = [[] for _ in range(len(name_list))]
            for i in judge_file.iterrows():
                line = i[1]
                for j in range(len(name_list)):
                    if line['source'] == name_list[j]:
                        sample_list[j].append(line['score'])

            acc_dict = {}
            correct_list = []

            # fine-grained
            for i in range(len(name_list)):
                correct_num = 0
                for j in sample_list[i]:
                    if j:
                        correct_num += 1
                correct_list.append(correct_num)
                acc = 100 * correct_num / len(sample_list[i])
                acc_dict[name_list[i]] = [acc]

            # 4 grained
            labela = ['zh', 'en']
            labelb = ['maths', 'physics']

            grain_list = [[x,y] for x in labela for y in labelb]
            for j in grain_list:
                dict_name = j[0] + "_" + j[1]
                correct_num = 0
                full_num = 0
                for i in range(len(name_list)):
                    if all(k in name_list[i] for k in j):
                        correct_num += correct_list[i]
                        full_num += len(sample_list[i])
                acc = 100 * correct_num / full_num
                acc_dict[dict_name] = [acc]

            # 2 grained
            grain_list = ['maths', 'physics']
            for j in grain_list:
                dict_name = j
                correct_num = 0
                full_num = 0
                for i in range(len(name_list)):
                    if j in name_list[i]:
                        correct_num += correct_list[i]
                        full_num += len(sample_list[i])
                acc = 100 * correct_num / full_num
                acc_dict[dict_name] = [acc]

            # AVG
            correct_num = sum(correct_list)
            acc = 100 * correct_num / len(judge_file)
            acc_dict['AVG'] = [acc]

            acc_pd = pd.DataFrame(acc_dict)
            acc_pd.to_csv(score_file, index=False, encoding='gbk')

        accdz = pd.read_csv(score_file)
        return accdz


class FlashMemory(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'FlashMemory_AUTOFILL': '/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/flash_memory_autofill_certificates.tsv',
        'FlashMemory_ENTITY':'/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/flash_memory_entity.tsv',
        'FlashMemory_STRUCT':'/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/flash_memory_struct_extract.tsv',
    }

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        if not 'model' in judge_kwargs:
            from .utils.kie import evaluate_with_pr
            model = 'Precision_Recall'
            suffix = eval_file.split('.')[-1]
            storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
            tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
            nproc = judge_kwargs.pop('nproc', 4)

            if not osp.exists(storage):
                # 读取预测数据 && GT数据
                data = load(eval_file)
                eval_results, summary = evaluate_with_pr(data)
                dump(eval_results, storage)
            score_pth = storage.replace('.xlsx', '_score.json')
            dump(summary, score_pth)
            return summary

        else:
            from .utils.kie import kie_auxeval, kie_acc
            model = judge_kwargs['model']
            suffix = eval_file.split('.')[-1]
            storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
            tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
            nproc = judge_kwargs.pop('nproc', 4)

            if not osp.exists(storage):
                data = load(eval_file)
                model = build_judge(max_tokens=1024, **judge_kwargs)
                assert model.working(), ('MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
                lt = len(data)
                lines = [data.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = [line['index'] for line in lines]

                ans = {}
                if osp.exists(tmp_file):
                    ans = load(tmp_file)
                tups = [x for x, i in zip(tups, indices) if i not in ans]
                indices = [i for i in indices if i not in ans]

                if len(indices):
                    new_results = track_progress_rich(
                        kie_auxeval,
                        tups,
                        nproc=nproc,
                        chunksize=nproc,
                        keys=indices,
                        save=tmp_file,
                    )
                    ans = load(tmp_file)
                    for k, v in zip(indices, new_results):
                        assert k in ans
                        assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

                data['res'] = [ans[idx]['res'] for idx in data['index']]
                data['log'] = [ans[idx]['log'] for idx in data['index']]
                dump(data, storage)

            score = kie_acc(storage)
            score_pth = storage.replace('.xlsx', '_score.csv')
            dump(score, score_pth)
            return score


class MemoryHouse(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MemoryHouse':'/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/memory_house.tsv'
    }

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.memory_house import memory_house_auxeval, memory_house_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=1024, **judge_kwargs)
            assert model.working(), ('MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    memory_house_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = memory_house_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score

class OCRBench_v2(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCRBench_v2':
        '/mnt/data/group/wangnan/code/VLMEvalKit20241026/LMUData/OCRBench_v2.tsv',
    }
    DATASET_MD5 = {'OCRBench_v2': '65d04fe07b4d4ee33e73fc8e7d4d46b0'}

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.ocrbrnch_v2_eval import process_predictions, ocrbench_v2_aggregate_accuracy, process_line
        from multiprocessing import Pool, cpu_count
        
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        predict_result = []

        with Pool(cpu_count()) as p:
            predict_result = list(tqdm(p.imap(process_line, lines), total=lt))

        res_data_list = process_predictions(predict_result)
        data['score'] = res_data_list
        dump(data, eval_file)
        en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(res_data_list)
        score_en_overall = sum(en_scores.values()) / len(en_scores)
        score_cn_overall = sum(cn_scores.values()) / len(cn_scores)
        final_score_dict = {**en_scores, **cn_scores}
        final_score_dict["English Overall Score"] = score_en_overall
        final_score_dict["Chinese Overall Score"] = score_cn_overall
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class LogicVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'LogicVista':
        'https://opencompass.openxlab.space/utils/VLMEval/LogicVista.tsv'
    }
    DATASET_MD5 = {'LogicVista': '41c5d33adf33765c399e0e6ae588c061'}

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.logicvista import LogicVista_auxeval, evaluate_logicvista

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in [
            'exact_matching', 'gpt-4-0125', 'gpt-4-turbo', 'gpt-4o-mini'
        ], model
        name_str_map = {
            'gpt-4-0125': 'gpt4',
            'gpt-4-turbo': 'gpt4-turbo',
            'gpt-4o-mini': 'gpt4o-mini'
        }
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn(
                    'OPENAI API is not working properly, will use exact matching for evaluation'
                )
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn(
                'OPENAI_API_KEY is not set properly, will use exact matching for evaluation'
            )
            model = None

        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{name_str}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{name_str}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage) and model is not None:
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'LogicVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    LogicVista_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res'] and ans[k]['hit'] == v['hit']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            data['hit'] = [ans[idx]['hit'] for idx in data['index']]

            dump(data, storage)
        if osp.exists(storage):
            accuracy_scores = evaluate_logicvista(storage)
            score_pth = storage.replace('.xlsx', '_score.csv')
            dump(accuracy_scores, score_pth)

            return accuracy_scores
