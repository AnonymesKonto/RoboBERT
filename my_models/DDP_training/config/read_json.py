import json

loaded_config = dict()
with open('DDP_training.json', 'r') as f:
    loaded_config = json.load(f)

print(loaded_config)