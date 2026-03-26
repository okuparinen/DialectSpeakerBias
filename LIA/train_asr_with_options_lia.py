
from datasets import load_dataset, Audio

from datasets import ClassLabel
import random
import pandas as pd
import sys
import os
import numpy as np

FOLDER=sys.argv[1]
INPUT=sys.argv[2]

if INPUT == 'orig':
	USE_VOICE_CONVERSION = False
	USE_MULTI_VC = False
	USE_AUGMENTATION = False
	USE_ORIGINAL = True

if INPUT == 'orig_aug':
	USE_VOICE_CONVERSION = False
	USE_MULTI_VC = False
	USE_AUGMENTATION = True
	USE_ORIGINAL = True

if INPUT == 'vc':
	USE_VOICE_CONVERSION = True
	USE_MULTI_VC = False
	USE_AUGMENTATION = False
	USE_ORIGINAL = False

if INPUT == 'multi_vc':
	USE_VOICE_CONVERSION = True
	USE_MULTI_VC = True
	USE_AUGMENTATION = False
	USE_ORIGINAL = False

if INPUT == 'orig_vc':
	USE_VOICE_CONVERSION = True
	USE_MULTI_VC = False
	USE_AUGMENTATION = False
	USE_ORIGINAL = True

if INPUT == 'orig_multi_vc':
	USE_VOICE_CONVERSION = True
	USE_MULTI_VC = True
	USE_AUGMENTATION = False
	USE_ORIGINAL = True

if INPUT not in ['orig', 'orig_aug', 'vc', 'multi_vc', 'orig_vc', 'orig_multi_vc']:
	raise Exception("The input value is not possible")

skn_train = pd.read_csv(f'speaker_partitions/{FOLDER}/train.csv', sep='\t', header=0)
skn_train = skn_train[skn_train['duration'] > 1.0]
skn_train = skn_train[skn_train['duration'] < 10.0]
skn_train = skn_train[:64100]
skn_dev = pd.read_csv(f'speaker_partitions/{FOLDER}/validation.csv', sep='\t', header=0)
skn_dev = skn_dev[skn_dev['duration'] > 1.0]
skn_dev = skn_dev[skn_dev['duration'] < 10.0]
skn_dev = skn_dev[:8180]

####
from datasets import Dataset
skn_train['path_data'] = 'data/' + skn_train['path'].astype(str)
train = Dataset.from_pandas(skn_train)
train = train.cast_column("path_data", Audio(sampling_rate=16000))
skn_dev['path_data'] = 'data/' + skn_dev['path'].astype(str)
dev = Dataset.from_pandas(skn_dev)
dev = dev.cast_column("path_data", Audio(sampling_rate=16000))

def extract_all_chars(batch):
  all_text = " ".join(item for item in batch["detailed"] if item and item.strip())
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_exists = os.path.isfile(f"LIA/ASR-orig-{FOLDER}/vocab.json")

if not vocab_exists:
    print('Vocab does not exist, creating it')
    os.makedirs(f'/scratch/project_2011201/LIA/ASR-orig-{FOLDER}', exist_ok=True)
    vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train.column_names)
    vocab_dev = dev.map(extract_all_chars, batched=True, remove_columns=dev.column_names)
    
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_dev["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    import json
    with open(f"LIA/ASR-orig-{FOLDER}/vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
    print(f"Vocab created in ASR-orig-{FOLDER}/vocab.json")

from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f"LIA/ASR-orig-{FOLDER}/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

import torch
import torchaudio

# Voice Conversion setup
if USE_VOICE_CONVERSION:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)

    def ref_generator(filestart):
        ref_paths = []
        for path, dir, files in os.walk("LIA/nordavinden"):
            for file in files:
                filepath = os.path.join(path, file)
                if file.startswith(filestart):
                    ref_paths.append(filepath)
        return knn_vc.get_matching_set(ref_paths)
    
    if USE_MULTI_VC:
        matching_sets = [
            ("vc1", ref_generator('nos1')),
            ("vc2", ref_generator('nos2')),
            ("vc3", ref_generator('nos3')),
            ("vc4", ref_generator('nos4'))
            ]
    else:
        matching_sets = [("vc1", ref_generator('nos1'))]
else:
    matching_sets = None

# Augmentation setup
if USE_AUGMENTATION:
    pitch_steps = [2, 4, -2, -4]
    pitch_transforms = [torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=s) for s in pitch_steps]
else:
    pitch_transforms = None

def safe_audio(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Audio safety guard:
      - Replace NaN/Inf with 0
      - Clip to [-1, 1]
      - Gentle RMS normalization to target_rms (~ -26 dBFS)
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    # Replace non-finite values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    # Hard clip
    np.clip(x, -1.0, 1.0, out=x)
    # Gentle RMS normalization
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    if rms > 0.0:
        scale = target_rms / rms
        # Avoid extreme scaling if rms is already healthy
        if 0.25 <= scale <= 4.0:
            x *= scale
    return x

import random

def prepare_dataset(batch, apply_vc=USE_VOICE_CONVERSION, apply_aug=USE_AUGMENTATION, vc_set=None):
    """
    Prepare different datasets one audio file at a time
    """

    audio = batch["path_data"]
    waveform = torch.tensor(audio["array"], dtype=torch.float).unsqueeze(0)

    # Voice Conversion
    if apply_vc and vc_set is not None:
        query_seq = knn_vc.get_features(waveform.to('cuda'))
        with torch.no_grad():
            converted_waveform = knn_vc.match(query_seq, vc_set, topk=4)
        waveform = converted_waveform.unsqueeze(0).cpu()

    # Augmentation
    if apply_aug and pitch_transforms:
        transform = random.choice(pitch_transforms)
        noise = torch.randn_like(waveform) * 0.005
        with torch.no_grad():
            shifted_waveform = transform(waveform)
            waveform = shifted_waveform + noise

    # Feature extraction
    batch["input_values"] = processor(waveform.squeeze().numpy(), sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    # Labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["detailed"]).input_ids

    return batch

def prepare_dataset_batched(batch, apply_vc=False, vc_set=None, apply_aug=False):
    """
    Batched version for preprocessing:
    - Handles multiple audio samples at once
    - Supports augmentation (CPU-friendly)
    - Voice conversion still single-process (GPU-heavy)
    """

    audio_arrays = [a["array"] for a in batch["path_data"]]
    sampling_rates = [a["sampling_rate"] for a in batch["path_data"]]

    if apply_vc and vc_set is not None:
        converted_arrays = []
        for array in audio_arrays:
            waveform = torch.tensor(array, dtype=torch.float).unsqueeze(0).to('cuda')
            query_seq = knn_vc.get_features(waveform)
            with torch.no_grad():
                converted_waveform = knn_vc.match(query_seq, vc_set, topk=4)
            converted_waveform = converted_waveform.squeeze().cpu().numpy()
            converted_waveform = safe_audio(converted_waveform)
            converted_arrays.append(converted_waveform)
        audio_arrays = converted_arrays

    if apply_aug and pitch_transforms:
        augmented_arrays = []
        for array in audio_arrays:
            waveform = torch.tensor(array, dtype=torch.float).unsqueeze(0)
            transform = random.choice(pitch_transforms)
            noise = torch.randn_like(waveform) * 0.005
            with torch.no_grad():
                shifted_waveform = transform(waveform)
                waveform = shifted_waveform + noise
            augmented_waveform = safe_audio(waveform.squeeze().numpy())
            augmented_arrays.append(augmented_waveform)
        audio_arrays = augmented_arrays

    # Feature extraction for batch
    audio_arrays = [safe_audio(np.asarray(a, dtype=np.float32)) for a in audio_arrays]
    inputs = processor(audio_arrays, sampling_rate=16000)
    with processor.as_target_processor():
        labels = processor(batch["detailed"]).input_ids

    return {
        "input_values": inputs.input_values,
        "input_length": [len(iv) for iv in inputs.input_values],
        "labels": labels
    }

# Original audio
if USE_ORIGINAL:
    train_ready = train.map(
        prepare_dataset_batched,
        remove_columns=train.column_names,
        num_proc=8,
        batched=True,
        batch_size=16
    )

# Voice conversion
if USE_VOICE_CONVERSION:
    train_ready_vc1 = train.map(lambda b: prepare_dataset_batched(b, apply_vc=True, vc_set=matching_sets[0][1]),
                                 remove_columns=train.column_names,
                                 batched=True,
                                 batch_size=6)

if USE_MULTI_VC:
    train_ready_vc2 = train.map(lambda b: prepare_dataset_batched(b, apply_vc=True, vc_set=matching_sets[1][1]),
                                 remove_columns=train.column_names,
                                 batched=True,
                                 batch_size=6)
    train_ready_vc3 = train.map(lambda b: prepare_dataset_batched(b, apply_vc=True, vc_set=matching_sets[2][1]),
                                 remove_columns=train.column_names,
                                 batched=True,
                                 batch_size=6)
    train_ready_vc4 = train.map(lambda b: prepare_dataset_batched(b, apply_vc=True, vc_set=matching_sets[3][1]),
                                 remove_columns=train.column_names,
                                 batched=True,
                                 batch_size=6)

# Augmentation
if USE_AUGMENTATION:
    train_ready_aug = train.map(
        lambda b: prepare_dataset_batched(b, apply_aug=True),
        remove_columns=train.column_names,
        num_proc=8,
        batched=True,
        batch_size=16
    )

dev_ready = dev.map(
        prepare_dataset_batched,
        remove_columns=dev.column_names,
        num_proc=8,
        batched=True,
        batch_size=16
    )

from datasets import concatenate_datasets

if INPUT == 'orig_aug':
    print('Concatenating train_ready and train_ready_aug')
    train_ready = concatenate_datasets([train_ready, train_ready_aug])

if INPUT == 'orig_multi_vc':
    print('Concatenating train_ready and train_ready_vc with multiple voices')
    train_ready = concatenate_datasets([train_ready, train_ready_vc1, train_ready_vc2, train_ready_vc3, train_ready_vc4])

if INPUT == 'multi_vc':
    print('Using only train_ready_vc with multiple voices')
    train_ready = concatenate_datasets([train_ready_vc1, train_ready_vc2, train_ready_vc3, train_ready_vc4])

if INPUT == 'orig_vc':
    print('Concatenating train_ready and train_ready_vc with one voice')
    train_ready = concatenate_datasets([train_ready, train_ready_vc1])

if INPUT == 'vc':
	print('Using only train_ready_vc with one voice')
	train_ready = train_ready_vc1

#########
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

#wer_metric = load_metric("wer")
from evaluate import load
import numpy as np
cer_metric = load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    #"GetmanY1/wav2vec2-large-fi-lp-cont-pt", 
    "facebook/mms-300m",
    attention_dropout=0.1, #0.09, #0.15
    hidden_dropout=0.1, #0.09, #0.1
    feat_proj_dropout=0.05, #0.05
    mask_time_prob=0.05, #0.05
    layerdrop=0.1, #0.092, #0.1
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_extractor()

from transformers import TrainingArguments, EarlyStoppingCallback, IntervalStrategy

training_args = TrainingArguments(
  output_dir=f"LIA/ASR-{INPUT}-{FOLDER}",
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=4,
  num_train_epochs=15,
  gradient_checkpointing=False,
  fp16=True,
  eval_strategy="epoch",
  save_strategy="epoch",
  logging_strategy="epoch",
  learning_rate=1e-5,
  warmup_steps=5000,
  max_grad_norm=1.0,
  save_total_limit=3,
  push_to_hub=False,
  metric_for_best_model = 'cer',
  greater_is_better=False,
  load_best_model_at_end=True,
  report_to="none",
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ready,
    eval_dataset=dev_ready,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
)

trainer.train()
