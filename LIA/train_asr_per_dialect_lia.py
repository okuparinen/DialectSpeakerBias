
# -*- coding: utf-8 -*-
from datasets import load_dataset, Audio, Dataset, concatenate_datasets
import random
import pandas as pd
import sys
import os
import numpy as np

FOLDER = sys.argv[1]
INPUT = sys.argv[2]

# --------------------------------------------------------------------
# MODES
# --------------------------------------------------------------------
if INPUT == 'orig':
    USE_VOICE_CONVERSION = False
    USE_MULTI_VC = False
    USE_AUGMENTATION = False
    USE_ORIGINAL = True
elif INPUT == 'orig_aug':
    USE_VOICE_CONVERSION = False
    USE_MULTI_VC = False
    USE_AUGMENTATION = True
    USE_ORIGINAL = True
elif INPUT == 'vc':
    USE_VOICE_CONVERSION = True
    USE_MULTI_VC = False
    USE_AUGMENTATION = False
    USE_ORIGINAL = False
elif INPUT == 'multi_vc':
    USE_VOICE_CONVERSION = True
    USE_MULTI_VC = True
    USE_AUGMENTATION = False
    USE_ORIGINAL = False
elif INPUT == 'orig_vc':
    USE_VOICE_CONVERSION = True
    USE_MULTI_VC = False
    USE_AUGMENTATION = False
    USE_ORIGINAL = True
elif INPUT == 'orig_multi_vc':
    USE_VOICE_CONVERSION = True
    USE_MULTI_VC = True
    USE_AUGMENTATION = False
    USE_ORIGINAL = True
elif INPUT == 'dialect_vc':
    USE_VOICE_CONVERSION = True
    USE_MULTI_VC = False
    USE_AUGMENTATION = False
    USE_ORIGINAL = False
else:
    raise Exception("The input value is not possible")

def _read_and_filter(path):
    df = pd.read_csv(path, sep='\t', header=0, encoding='utf-8')
    df = df[df['duration'] > 1.0]
    df = df[df['duration'] < 20.0]
    df = df[:8180]
    return df

if INPUT != 'dialect_vc':
    skn_train = _read_and_filter(f'speaker_partitions/{FOLDER}/train.csv')
else:
    DIALECT_FILES = {
        "Midlandsk":           "Midlandsk_train.csv",
        "Namdal-Uttrøndersk":  "Namdal-Uttrøndersk_train.csv",
        "Nordland-Helgeland":  "Nordland-Helgeland_train.csv",
        "Nordvestlandsk":      "Nordvestlandsk_train.csv",
        "Sørlandsk":           "Sørlandsk_train.csv",
        "Sørvestlandsk":       "Sørvestlandsk_train.csv",
        "Troms-Finnmarks-mål": "Troms-Finnmarks-mål_train.csv",
        "Østlandsk":           "Østlandsk_train.csv",
        "Østtrøndsk":          "Østtrøndsk_train.csv",
    }
    train_dfs = []
    for area, fname in DIALECT_FILES.items():
        fpath = f'speaker_partitions/{FOLDER}/{fname}'
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Expected per-dialect train file not found: {fpath}")
        df = _read_and_filter(fpath)
        df['dialect_area'] = area
        train_dfs.append(df)
    skn_train = pd.concat(train_dfs, ignore_index=True)

skn_dev = _read_and_filter(f'speaker_partitions/{FOLDER}/validation.csv')

from datasets import Dataset

def extract_all_chars(batch):
    all_text = " ".join(item for item in batch["detailed"] if item and item.strip())
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_exists = os.path.isfile(f"LIA/ASR-orig-{FOLDER}/vocab.json")
if not vocab_exists:
    print('Vocab does not exist, creating it')
    os.makedirs(f'LIA/ASR-orig-{FOLDER}', exist_ok=True)

    vocab_train = Dataset.from_pandas(skn_train[['detailed']].copy())
    vocab_dev = Dataset.from_pandas(skn_dev[['detailed']].copy())

    vocab_train = vocab_train.map(extract_all_chars, batched=True, batch_size=-1,
                                  keep_in_memory=True, remove_columns=vocab_train.column_names)
    vocab_dev = vocab_dev.map(extract_all_chars, batched=True, remove_columns=vocab_dev.column_names)

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

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    f"LIA/ASR-orig-{FOLDER}/",
    unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0,
    do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

import torch
import torchaudio

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

    VOICE_TO_REFSTART = {
        "vc1": "nos1",
        "vc2": "nos2",
        "vc3": "nos3",
        "vc4": "nos4",
        "vc5": "nos5",
        "vc6": "nos6",
        "vc7": "nos7",
        "vc8": "nos8",
        "vc9": "nos9",
    }

    def build_matching_sets(voices_needed):
        ms = {}
        for v in sorted(set(voices_needed)):
            if v not in VOICE_TO_REFSTART or VOICE_TO_REFSTART[v].startswith("TODO"):
                raise ValueError(
                    f"Missing reference start for {v}. Please set VOICE_TO_REFSTART['{v}'] to your reference prefix."
                )
            ms[v] = ref_generator(VOICE_TO_REFSTART[v])
        return ms

else:
    knn_vc = None

if USE_AUGMENTATION:
    pitch_steps = [2, 4, -2, -4]
    pitch_transforms = [torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=s) for s in pitch_steps]
else:
    pitch_transforms = None

def safe_audio(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Audio safety guard for audio issues:
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
def prepare_dataset_batched(batch, apply_vc=False, vc_set=None, apply_aug=False):
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
            augmented = safe_audio(waveform.squeeze().numpy())
            augmented_arrays.append(augmented)
        audio_arrays = augmented_arrays
    audio_arrays = [safe_audio(np.asarray(a, dtype=np.float32)) for a in audio_arrays]
    inputs = processor(audio_arrays, sampling_rate=16000)
    with processor.as_target_processor():
        labels = processor(batch["detailed"]).input_ids
    return {
        "input_values": inputs.input_values,
        "input_length": [len(iv) for iv in inputs.input_values],
        "labels": labels
    }

skn_dev['path_data'] = 'data/' + skn_dev['path'].astype(str)
dev = Dataset.from_pandas(skn_dev)
dev = dev.cast_column("path_data", Audio(sampling_rate=16000))
dev_ready = dev.map(
    prepare_dataset_batched,
    remove_columns=dev.column_names,
    num_proc=1,
    batched=True,
    batch_size=32
)

if INPUT == 'dialect_vc':
    # Voice mapping (exactly one voice per dialect)
    AREA_TO_VOICE = {
        "Midlandsk":                   "vc9",
        "Namdal-Uttrøndersk":          "vc1",
        "Nordland-Helgeland":          "vc4",
        "Nordvestlandsk":              "vc8",
        "Sørlandsk":                   "vc3",
        "Sørvestlandsk":               "vc5",
        "Troms-Finnmarks-mål":         "vc7",
        "Østlandsk":                   "vc2",
        "Østtrøndsk":                  "vc6",
    }

    voices_needed = list(AREA_TO_VOICE.values())
    matching_sets_dict = build_matching_sets(voices_needed)

    train_ready_parts = []
    for area, voice in AREA_TO_VOICE.items():
        area_df = skn_train[skn_train['dialect_area'] == area].copy()
        if len(area_df) == 0:
            print(f"[WARN] No rows for area '{area}' in input CSVs. Skipping.")
            continue
        area_df['path_data'] = 'data/' + area_df['path'].astype(str)
        area_ds = Dataset.from_pandas(area_df)
        area_ds = area_ds.cast_column("path_data", Audio(sampling_rate=16000))
        print(f"Applying VC '{voice}' to area '{area}' with {len(area_ds)} samples...")
        area_ready = area_ds.map(
            lambda b, _vc=matching_sets_dict[voice]: prepare_dataset_batched(b, apply_vc=True, vc_set=_vc),
            remove_columns=area_ds.column_names,
            batched=True,
            batch_size=8
        )
        train_ready_parts.append(area_ready)

    if len(train_ready_parts) == 0:
        raise RuntimeError("No per-dialect training data prepared. Check your *_train.csv files.")

    train_ready = concatenate_datasets(train_ready_parts).shuffle(seed=42)

# Not designed to be used, use with_options instead
else:
    skn_train['path_data'] = 'data/' + skn_train['path'].astype(str)
    train = Dataset.from_pandas(skn_train)
    train = train.cast_column("path_data", Audio(sampling_rate=16000))

    if USE_VOICE_CONVERSION:
        def ref_generator(filestart):
            ref_paths = []
            for path, dir, files in os.walk("/scratch/project_2011201/SKN/la_data"):
                for file in files:
                    filepath = os.path.join(path, file)
                    if file.startswith(filestart) and os.path.getsize(filepath) > 1000000:
                        ref_paths.append(filepath)
            return knn_vc.get_matching_set(ref_paths[:30])

        if USE_MULTI_VC:
            matching_sets = [
                ("vc1", ref_generator('hame_hat1')),
                ("vc2", ref_generator('sate_koke1')),
                ("vc3", ref_generator('pohp_yii1')),
                ("vc4", ref_generator('kare_luu1')),
            ]
        else:
            matching_sets = [("vc1", ref_generator('hame_hat1'))]
    else:
        matching_sets = None

    # Augmentation
    if USE_AUGMENTATION:
        train_ready = train.map(
            lambda b: prepare_dataset_batched(b, apply_aug=True),
            remove_columns=train.column_names,
            num_proc=8,
            batched=True,
            batch_size=32
        )
    elif USE_ORIGINAL and not USE_VOICE_CONVERSION:
        train_ready = train.map(
            prepare_dataset_batched,
            remove_columns=train.column_names,
            num_proc=8,
            batched=True,
            batch_size=32
        )
    else:
        # VC branches
        if USE_VOICE_CONVERSION and not USE_MULTI_VC:
            train_ready_vc1 = train.map(
                lambda b: prepare_dataset_batched(b, apply_vc=True, vc_set=matching_sets[0][1]),
                remove_columns=train.column_names, batched=True, batch_size=8
            )
            train_ready = train_ready_vc1 if not USE_ORIGINAL else concatenate_datasets([train_ready, train_ready_vc1])
        elif USE_VOICE_CONVERSION and USE_MULTI_VC:
            train_ready_vc = []
            for _, vcset in matching_sets:
                part = train.map(
                    lambda b, _vc=vcset: prepare_dataset_batched(b, apply_vc=True, vc_set=_vc),
                    remove_columns=train.column_names, batched=True, batch_size=8
                )
                train_ready_vc.append(part)
            if USE_ORIGINAL:
                orig_ready = train.map(
                    prepare_dataset_batched,
                    remove_columns=train.column_names,
                    num_proc=8,
                    batched=True, batch_size=32
                )
                train_ready = concatenate_datasets([orig_ready] + train_ready_vc)
            else:
                train_ready = concatenate_datasets(train_ready_vc)

from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
from evaluate import load

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

cer_metric = load("cer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained(
    # "GetmanY1/wav2vec2-large-fi-lp-cont-pt",
    "facebook/mms-300m",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.05,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
model.freeze_feature_extractor()

from transformers import TrainingArguments, Trainer
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
    learning_rate=1e-4,
    warmup_steps=5000,
    max_grad_norm=1.0,
    save_total_limit=3,
    push_to_hub=False,
    metric_for_best_model='cer',
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ready,
    eval_dataset=dev_ready,
    tokenizer=processor.feature_extractor
)

trainer.train()

