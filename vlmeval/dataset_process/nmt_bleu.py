from tqdm import tqdm
import sacrebleu
import ast
import pandas as pd
import os
from collections import defaultdict
import json
import sys


eval_file = sys.argv[1]
model_dir = os.path.dirname(eval_file)
model, data_name = eval_file.split('/')[-2], os.path.basename(eval_file).split('.xlsx')[0]
output_path = os.path.join(model_dir, data_name)


data = pd.read_excel(eval_file)
lt = len(data)
all_lines = [data.iloc[i] for i in range(lt)]

TranslationBench_items = defaultdict(list)

for i in range(len(all_lines)):
    line = all_lines[i]
    translation_language = line['type']
    TranslationBench_items[translation_language].append({'input': line['question'], 'output': line['answers'], 'pred': line['prediction']})

for translation_language, lines in TranslationBench_items.items():
    num, score = 0, 0
    result, low_bleu = [], []
    query = ''
    outfile = os.path.join(output_path, 'bleu_result', f'{translation_language}.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    for js in lines:
        num += 1

        input = js['input']
        predict = js['pred']
        answers = js['output']

        tgtlang = translation_language.split('-')[-1] # 翻译成的语种
        tokenize = 'none'
        if tgtlang == 'ko':
            tokenize = 'ko-mecab'
        elif tgtlang == 'ja':
            tokenize = 'ja-mecab'
        elif tgtlang == 'zh':
            tokenize = 'zh'
        else:
            tokenize = 'none'
        bleu = sacrebleu.corpus_bleu([predict], [[answers]], tokenize=tokenize).score

        js['bleu'] = bleu
        score += bleu

        if bleu < 10:
            low_bleu.append(js)
        else:
            result.append(js)
        
    score = score / num
    print(f"translation_language:{translation_language}, Avg bleu: {score}")

    info = []
    info.append({"task":"文档翻译","prompt":query, 'model':model, "total_num":len(lines), 'avg_bleu':score, 'low_bleu':low_bleu, 'high_bleu': result})
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
