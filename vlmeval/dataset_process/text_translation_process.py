import json
import pandas as pd
import os


def json2tsv(json_file_list, tsv_file):
    data = {"dataset_name": [],
            "id": [],
            "question": [],
            "answers": [],
            "type": [],
            "index": [],
            "answer": []
            }

    start_idx = 0
    for json_file in json_file_list:
        translation_language = os.path.basename(json_file).split('.')[0]
        dataset_name = 'translation_' + translation_language

        with open(json_file, 'r') as file:
            info = json.load(file)

        for idx, item in enumerate(info):
            data["dataset_name"].append(dataset_name)
            data["id"].append(start_idx + idx)
            data['question'].append(item['instruction'] + item['input'])
            data['answers'].append(item['output'])
            data['type'].append(translation_language)
            data['index'].append(start_idx + idx)
            data['answer'].append([item['output']])
        start_idx += len(info)

    df = pd.DataFrame(data)
    df.to_csv(tsv_file, sep='\t', index=False)


if __name__ == '__main__':
    json_file_list = ['/mnt/data/group/vlm_data/business_scenario_data/translation_data/testset/flores200-no-tag/en-zh.json',
                      '/mnt/data/group/vlm_data/business_scenario_data/translation_data/testset/flores200-no-tag/zh-en.json']  
    tsv_file = '/mnt/data/group/vlm_data/business_scenario_data/translation_data/testset/flores200-no-tag/flores200-no-tag.tsv'
    json2tsv(json_file_list, tsv_file)