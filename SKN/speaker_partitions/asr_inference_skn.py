from datasets import load_dataset, Audio

from datasets import ClassLabel
import random
import pandas as pd
import sys

FOLDER=sys.argv[1]
METHOD=sys.argv[2]
INPUT=sys.argv[3]

skn_test = pd.read_csv(f'speaker_partitions/{FOLDER}/{INPUT}.csv', sep='\t', header=0)
skn_test = skn_test.dropna(how='any', axis=0)
skn_test = skn_test[skn_test['duration'] > 1.0]
skn_test = skn_test[skn_test['duration'] < 20.0]

from datasets import Dataset
skn_test['path_data'] = 'data/' + skn_test['path'].astype(str)
test = Dataset.from_pandas(skn_test)
test = test.cast_column("path_data", Audio(sampling_rate=16000))

from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f"/scratch/project_2011201/SKN/ASR-orig-{FOLDER}/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
    audio = batch["path_data"]
#    text = batch["big_clean"]
    
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
#    batch["labels"] = processor(text=text).input_ids
    with processor.as_target_processor():
        batch["labels"] = processor(batch["detailed"]).input_ids
    
    return batch

test_ready = test.map(prepare_dataset, remove_columns=test.column_names)

import os
import re
checkpoints = [entry for entry in os.listdir(f'ASR-{METHOD}-{FOLDER}') if entry.startswith("checkpoint")]
values = [re.sub('checkpoint-', '', item) for item in checkpoints]
values = [int(value) for value in values]
checkpoint_index = values.index(max(values))
last_checkpoint = checkpoints[checkpoint_index]

import torch

model = Wav2Vec2ForCTC.from_pretrained(f"ASR-{METHOD}-{FOLDER}/{last_checkpoint}").to("cuda")

length = len(test)
predictions = []
references = []

for i in range(0, length, 1):
    input_dict = processor(test_ready[i]["input_values"], return_tensors="pt", padding=True)
    logits = model(input_dict.input_values.to("cuda")).logits
    
    pred_ids = torch.argmax(logits, dim=-1)[0]
    
    prediction = processor.decode(pred_ids)
    predictions.append(prediction)

    reference = test[i]["detailed"]
    references.append(reference)

with open(f"ASR-{METHOD}-{FOLDER}/{INPUT}_predictions.txt", "w") as f_pred:
    for line in predictions:
        f_pred.write(line + '\n')

with open(f"ASR-{METHOD}-{FOLDER}/{INPUT}_references.txt", "w") as f_ref:
    for line in references:
        f_ref.write(line + '\n')

