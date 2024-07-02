# -*- coding: utf-8 -*-
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets, DatasetDict
import json, copy

### Map functions
def add_text_as_main_target_use_2feature(example, concat_key1, concat_key2):
    """ textとして，複数のfeatureから構成される場合にテキストとして結合 """
    example['text'] = f"{example[concat_key1]}\n{example[concat_key2]}"
    return example

def cp_value(example, cp_from_key, cp_to_key):
    """ key-valueを指定のkeyとしてコピー """
    example[cp_to_key] = f"{example[cp_from_key]}"
    return example

def add_key_val(example, tgt_key, tgt_val):
    example[tgt_key] = tgt_val
    return example

### Main ##################################################################################################################################

def dl_peS2o():
    ds_peS2o = load_dataset("allenai/peS2o")['train']  # textのみ

    print(ds_peS2o[0])
    print(len(ds_peS2o))

    # filter
    ds_peS2o = ds_peS2o.filter(lambda example: example['source'] == 's2orc/train' and example['version'] == 'v2')
    print('filtered')
    print(ds_peS2o[0])
    print(len(ds_peS2o))

    remove_features = list(set(ds_peS2o[0].keys()) - set(['text']))
    ds_peS2o = ds_peS2o.remove_columns(remove_features)
    print('remove column')
    print(ds_peS2o[0])
    # save
    print('saving... ')
    save_path = "/storage6/peS2o_s2orcV2_en.jsonl"
    ds_peS2o.to_json(save_path, force_ascii=False)

def dl_orcaMath():
    #　変換方針: (original-key) question, answer -> text
    ds_orcaMath = load_dataset('microsoft/orca-math-word-problems-200k')['train']
    print(ds_orcaMath[0])
    print(len(ds_orcaMath))
    ds_orcaMath = ds_orcaMath.map(add_text_as_main_target_use_2feature, fn_kwargs={'concat_key1': 'question', 'concat_key2': 'answer'})
    print(ds_orcaMath[0]['text'])
    remove_features = list(set(ds_orcaMath[0].keys()) - set(['text']))
    ds_orcaMath = ds_orcaMath.remove_columns(remove_features)
    print(ds_orcaMath[0])
    ds_orcaMath.to_json("/storage6/corpus/category/MATH/raw/OrcaMathWordProblems/orca-math-word-problems_text_EN.jsonl")

def dl_metaMath():
    # 変換方針 original_question, resnponse -> text
    ds_MetaMath = load_dataset('agicorp/MetaMathQA')['train']
    print(ds_MetaMath[0])
    print(len(ds_MetaMath))
    ds_MetaMath = ds_MetaMath.map(add_text_as_main_target_use_2feature, fn_kwargs={'concat_key1': 'original_question', 'concat_key2': 'response'})
    print(ds_MetaMath[0]['text'])
    remove_features = list(set(ds_MetaMath[0].keys()) - set(['text']))
    ds_MetaMath = ds_MetaMath.remove_columns(remove_features)
    print(ds_MetaMath[0])
    ds_MetaMath.to_json("/storage6/corpus/category/MATH/raw/MetaMath/MetaMathQA_text_EN.jsonl")

def dl_atlasMath():
    # 変換方針 instruction, output -> text
    ds_atlasMath = load_dataset('AtlasUnified/atlas-math-sets')['train']
    print(ds_atlasMath[0])
    print(f"len: {len(ds_atlasMath)}")
    ds_atlasMath = ds_atlasMath.map(add_text_as_main_target_use_2feature, fn_kwargs={'concat_key1': 'instruction', 'concat_key2': 'output'})
    print(ds_atlasMath[0]['text'])
    remove_features = list(set(ds_atlasMath[0].keys()) - set(['text']))
    ds_atlasMath = ds_atlasMath.remove_columns(remove_features)
    print(ds_atlasMath[0])
    ds_atlasMath.to_json("/storage6/corpus/category/MATH/raw/AtlasMathSets/AtlasMathSets_text_jsonl")

def dl_basicMath():
    # 変換方針 instruction, output -> text
    _ds = load_dataset('lmlab/basic-math-10m')['train']
    print(_ds[0])
    print(f"len: {len(_ds)}")
    _ds = _ds.map(add_text_as_main_target_use_2feature, fn_kwargs={'concat_key1': 'instruction', 'concat_key2': 'answer'})
    print(_ds[0]['text'])
    remove_features = list(set(_ds[0].keys()) - set(['text']))
    _ds = _ds.remove_columns(remove_features)
    print(_ds[0])
    _ds.to_json("/storage6/corpus/category/MATH/raw/basicMath/basicMath_10m_jsonl")

def dl_conceptKG():
    _ds = load_dataset('RJZ/ConceptNetSyntheticPhi3Text_ja')['train']
    print(_ds[0])
    print(f"len: {len(_ds)}")
    _ds.to_json("/storage6/jiez/kg_synth/conceptnet_triples_phi3text_ja.jsonl")



### SFT ##################################################################################################################################
def dl_wikiQA_ja():
    def format_wikiQA(example):
        # JSQuAD format: https://techblog.yahoo.co.jp/entry/2022122030379907/
        text = f"[タイトル] {example['title']}\n{example['text']}\n\n質問: {example['query']}\n答え: {example['answer']}"
        example['text'] = text
        return example

    _ds = load_dataset('cl-nagoya/auto-wiki-qa')['train']
    print(_ds[0])
    print(f"len: {len(_ds)}")
    _ds = _ds.map(format_wikiQA)
    print(_ds[0]['text'])
    remove_features = list(set(_ds[0].keys()) - set(['text']))
    _ds = _ds.remove_columns(remove_features)
    print(_ds[0])
    _ds.to_json("/storage6/corpus/SFT/AutoWikiQA_ja_SFTtext.jsonl", force_ascii=False)

def dl_xP3x_ja():
    # inputs, targets -> (before_template) instruction="", input, response
    jpn_datasets = []
    for lang in ["jpn_Kana","jpn_Hani", "jpn_Hira", "jpn_Jpan"]: # Japanese (Katakana), Japanese (Kanji), Japanese (Hiragana), Japanese
        print(f"lang -> {lang}")
        _ds = load_dataset("CohereForAI/xP3x", lang)['train'] # Japanese (Kanji)
        print(_ds[0])
        print(len(_ds))
        _ds = _ds.map(cp_value, fn_kwargs={'cp_from_key': 'inputs', 'cp_to_key': 'input'})
        _ds = _ds.map(cp_value, fn_kwargs={'cp_from_key': 'targets', 'cp_to_key': 'response'})
        _ds = _ds.map(add_key_val, fn_kwargs={'tgt_key': 'instruction', 'tgt_val': ""})
        _ds = _ds.remove_columns(list(set(_ds.features) - set(['instruction','input', 'response'])))
        jpn_datasets.append(_ds)

    jpn_ds = concatenate_datasets(jpn_datasets)
    print(len(jpn_ds))
    print(jpn_ds[4])
    jpn_ds.to_json("/storage6/corpus/SFT/xP3x_ja.jsonl", force_ascii=False)

def map_ja_SFT_format(example):
    _instruction = example['instruction']
    _input = example['input']
    _response = example['response']
    text = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{_instruction}\n\n### 入力:\n{_input}\n\n### 応答:\n{_response}"
    example['text'] = text
    return example

def convert_xP3x_ja():
    # SFTフォーマットのtextを作成
    ja_path = "/storage6/corpus/SFT/xP3x_ja.jsonl"
    _ds = load_dataset("json", data_files=ja_path)['train']
    _ds = _ds.map(map_ja_SFT_format)
    remove_features = list(set(_ds[0].keys()) - set(['text']))
    _ds = _ds.remove_columns(remove_features)
    _ds.to_json("/storage6/corpus/SFT/xP3x_ja_SFTtext.jsonl", force_ascii=False)


def dl_xP3x_en():
    _ds = load_dataset("CohereForAI/xP3x", "eng_Latn")['train']
    print(_ds[0])
    print(len(_ds))
    _ds = _ds.map(cp_value, fn_kwargs={'cp_from_key': 'inputs', 'cp_to_key': 'input'})
    _ds = _ds.map(cp_value, fn_kwargs={'cp_from_key': 'targets', 'cp_to_key': 'response'})
    _ds = _ds.map(add_key_val, fn_kwargs={'tgt_key': 'instruction', 'tgt_val': ""})
    _ds = _ds.remove_columns(list(set(_ds.features) - set(['instruction','input', 'response'])))
    _ds.to_json("/storage6/corpus/SFT/xP3x_en.jsonl", force_ascii=False)

def map_en_SFT_format(example):
    _instruction = example['instruction']
    _input = example['input']
    _response = example['response']
    text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{_instruction}\n\n### Input:\n{_input}\n\n### Response:\n{_response}"
    example['text'] = text
    return example

def convert_xP3x_en():
    # SFTフォーマットのtextを作成
    ja_path = "/storage6/corpus/SFT/xP3x_en.jsonl"
    _ds = load_dataset("json", data_files=ja_path)['train']
    _ds = _ds.map(map_en_SFT_format)
    remove_features = list(set(_ds[0].keys()) - set(['text']))
    _ds = _ds.remove_columns(remove_features)
    _ds.to_json("/storage6/corpus/SFT/xP3x_en_SFTtext.jsonl", force_ascii=False)

def dl_guanaco_ja():
    # instruction, input, output -> (before_template) instruction, input, response
    _ds = load_dataset("fujiki/guanaco_ja")['train']
    print(_ds[0])
    print(len(_ds))
    _ds = _ds.map(cp_value, fn_kwargs={'cp_from_key': 'output', 'cp_to_key': 'response'})
    _ds = _ds.remove_columns(list(set(_ds.features) - set(['instruction','input', 'response'])))
    _ds.to_json("/storage6/corpus/SFT/guanaco_ja.jsonl", force_ascii=False)

def convert_coTangent_ja():
    with open('./CoTangent_separated_ja.json') as f:
        obj_list = json.load(f)
    #print(obj_list)

    # output jsonl
    with open('./conv_CoTangent_ja.jsonl', mode='w', encoding="utf-8") as out:
        for _obj in obj_list:
            new_obj = copy.deepcopy(_obj)
            new_obj["response"] = f"{_obj['cot']}\nしたがって、{_obj['output']}"
            del new_obj['cot'], new_obj['output']
            enc = json.dumps(new_obj, ensure_ascii=False)
            out.write(f"{enc}\n")
    print('output ./conv_CoTangent_ja.jsonl')

def add_text_coTangent_ja():
    cot_path = "/storage6/corpus/SFT/conv_CoTangent_ja.jsonl"
    _ds = load_dataset("json", data_files=cot_path)['train']

    def map_concatenated_text(example):
        _txt = example['instruction']+'\n' if example['instruction'] != "" else ""
        _txt += example['input']+'\n' if example['input'] != "" else ""
        _txt += example['response'] if example['response'] != "" else ""
        example['text'] = _txt
        return example
    _ds = _ds.map(map_concatenated_text)
    _ds.to_json("/storage6/corpus/SFT/ja_sft_CoTangent_text.jsonl", force_ascii=False)    


def dl_llmJapanese():
    # instruction, input, output -> (before_template) instruction, input, response
    _ds = load_dataset("izumi-lab/llm-japanese-dataset")['train']
    print(_ds[0])
    print(len(_ds))
    _ds = _ds.map(cp_value, fn_kwargs={'cp_from_key': 'output', 'cp_to_key': 'response'})
    _ds = _ds.remove_columns(list(set(_ds.features) - set(['instruction','input', 'response'])))
    _ds.to_json("/storage6/corpus/SFT/llm_japanese.jsonl", force_ascii=False)

def add_text_llmJapanese():
    # `text`=instruction +\n+ input +\n+ response
    llm_path = "/storage6/corpus/SFT/llm_japanese.jsonl"
    _ds = load_dataset("json", data_files=llm_path)['train']

    def map_concatenated_text(example):
        example['text'] = f"{example['instruction']}\n{example['input']}\n{example['response']}"
        return example
    _ds = _ds.map(map_concatenated_text)
    _ds.to_json("/storage6/corpus/SFT/sft_llm_japanese_text.jsonl", force_ascii=False)

if __name__ == "__main__":
    #dl_peS2o()
    #dl_orcaMath()
    #dl_metaMath()
    #dl_atlasMath()
    #dl_basicMath()

    ### SFT
    dl_wikiQA_ja()
    #dl_xP3x_ja()
    #convert_xP3x_ja()
    #dl_xP3x_en()
    #convert_xP3x_en()
    #dl_guanaco_ja()
    #convert_coTangent_ja()
    #dl_llmJapanese()