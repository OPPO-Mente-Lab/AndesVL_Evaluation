import os
import json
import glob
import sys


eval_file = sys.argv[1]
model_dir, data_name = os.path.dirname(eval_file), os.path.basename(eval_file).split('.xlsx')[0]
results_dir = os.path.join(model_dir, f'{data_name}/bleu_result_new')
files = [file for file in os.listdir(results_dir) if file.endswith('.json')]

TRANSLATION_LANGUAGES = ['en-zh', 'zh-en', 'en-ko', 'ko-en', 'en-id', 'id-en', 'en-hi', 'hi-en', 'en-es', 'es-en', 'en-pt', 'pt-en', 'en-th', 'th-en', \
    'en-ms', 'ms-en', 'en-it', 'it-en', 'en-fr', 'fr-en', 'en-vi', 'vi-en','en-ru', 'ru-en', 'zh-de', 'de-zh', 'zh-ja', 'ja-zh', 'zh-ko', 'ko-zh', \
        'zh-id', 'id-zh', 'zh-hi', 'hi-zh', 'zh-es', 'es-zh', 'zh-pt', 'pt-zh', 'zh-th', 'th-zh', 'zh-ms', 'ms-zh', 'zh-it', 'it-zh', 'zh-fr', 'fr-zh', \
            'zh-vi', 'vi-zh', 'zh-ru', 'ru-zh']

results = {}
for file in files:
    language = file.split('.')[0]
    with open(os.path.join(results_dir, file), 'r') as f:
        js = json.load(f)
    new_bleu = js[0]['pipe_bleu']
    results[language] = new_bleu

revise_results = {}
for language in TRANSLATION_LANGUAGES:
    if language in results:
        revise_results[language] = results[language]

revise_results['avg'] = sum(revise_results.values()) / len(revise_results.keys())

save_path = os.path.join(model_dir, f'{data_name}', f'{data_name}_NMT_score.json')
print(f"最新的结果保存在{save_path}")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(revise_results, f, ensure_ascii=False, indent=4)
 

