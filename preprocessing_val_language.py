from omegaconf import OmegaConf
from transformers import BertModel, BertTokenizer
import open_clip

def preprocessing_val_languages_bert(val_annotations):
    import json

    loaded_config = dict()
    with open('config_path.json', 'r') as f:
        loaded_config = json.load(f)
    BERT_PATH = loaded_config["BERT_path"]

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    bert = BertModel.from_pretrained(BERT_PATH)
    new_val_annotations = dict()
    for task, text in val_annotations.items():
        text = [task.replace("_"," "), text[0]]
        tokens_info = tokenizer(text,
                        padding="max_length",
                        max_length=16,
                        truncation=True,
                        return_tensors="pt"
                        )
        languages_x_dict = bert(input_ids=tokens_info["input_ids"], attention_mask=tokens_info["attention_mask"]) 
        languages_x = languages_x_dict["last_hidden_state"].detach()
        tokens_mask = tokens_info["attention_mask"]
        new_val_annotations[task] = {"text": text, "languages_x": languages_x}

    return new_val_annotations
    
def preprocessing_val_languages_clip(val_annotations):
    # create text encoder
    clip_vision_encoder_path: str = "ViT-L-14"
    clip_vision_encoder_pretrained: str = "openai"
    clip_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    text_encoder = clip_encoder

    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    new_val_annotations = dict()
    for task, text in val_annotations.items():
        text = [task.replace("_"," "), text[0]]
        tokens_info = tokenizer(text)
        languages_x = text_encoder.encode_text(tokens_info).unsqueeze(1).detach()
        new_val_annotations[task] = {"text": text, "languages_x": languages_x}

    return new_val_annotations
    


if __name__ == "__main__":
    val_annotations = OmegaConf.load("./calvin_models/conf/annotations/new_playtable_validation.yaml")
    print(preprocessing_val_languages(val_annotations))
