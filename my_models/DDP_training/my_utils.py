def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def lang_to_embed(selected_tasks:dict) -> dict:
    import torch
    from transformers import BertModel, BertTokenizer

    BERT_PATH = "./bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    bert = BertModel.from_pretrained(BERT_PATH)

    tasks_lang_x = dict()
    tasks_lang_mask = dict()

    for task in selected_tasks.keys():
        task_str = task.replace("_"," ")
        tasks_lang_x[task_str] = "no tensor"

    input_str = list(tasks_lang_x.keys())
    tokens_info = tokenizer(input_str,
                        padding="max_length",
                        max_length=8,
                        truncation=True,
                        return_tensors="pt"
                        )

    languages_x = bert(input_ids=tokens_info["input_ids"], attention_mask=tokens_info["attention_mask"])["last_hidden_state"].detach() 
    tokens_mask = 1 - tokens_info["attention_mask"]
    tokens_mask = torch.where(tokens_mask == 1, -torch.inf, tokens_mask).detach()

    for name_id, task_str in enumerate(input_str):
        tasks_lang_x[task_str] = languages_x[name_id]
        tasks_lang_mask[task_str] = tokens_mask[name_id]

    return tasks_lang_x, tasks_lang_mask
    
def remove_prefix(storage, isEval=False):
    from collections import OrderedDict
    # 创建一个新的有序字典
    new_state_dict = OrderedDict()
    # 遍历状态字典中的键和值
    for k, v in storage.items():
        # 去除键中的"module."前缀
        name = k.replace("module.", "")
        if isEval == True:
           name = name.replace("nets.", "")
        # 将新的键和值添加到新的字典中
        new_state_dict[name] = v
    # 返回新的存储位置和新的字典
    return new_state_dict

