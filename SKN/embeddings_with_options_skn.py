from datasets import load_dataset, Audio
#from datasets import ClassLabel, Sequence
import random
import pandas as pd
import soundfile as sf
import sys

FOLDER=sys.argv[1]
USE_VOICE_CONVERSION = False  # Toggle VC globally
USE_AUGMENTATION = False

skn_train = pd.read_csv(f'speaker_partitions/{FOLDER}/train.csv', sep='\t', header=0)
skn_train = skn_train.dropna(how='any', axis=0)
skn_train = skn_train[skn_train['duration'] > 2.0]
skn_dev = pd.read_csv(f'speaker_partitions/{FOLDER}/validation.csv', sep='\t', header=0)
skn_dev = skn_dev.dropna(how='any', axis=0)
skn_dev = skn_dev[skn_dev['duration'] > 2.0]
skn_test = pd.read_csv(f'speaker_partitions/{FOLDER}/test.csv', sep='\t', header=0)
skn_test = skn_test.dropna(how='any', axis=0)
skn_test = skn_test[skn_test['duration'] > 2.0]

from datasets import Dataset
skn_train['path_data'] = 'data/' + skn_train['path'].astype(str)
train = Dataset.from_pandas(skn_train)
train = train.remove_columns(["Unnamed: 0", "SpeakerID", "starttime", "duration", "original", "path", "DocID", "role", "Location", "detailed", "simple", "__index_level_0__", "MurrealueOlli"])
train = train.cast_column("path_data", Audio(sampling_rate=16000))

skn_dev['path_data'] = 'data/' + skn_dev['path'].astype(str)
dev = Dataset.from_pandas(skn_dev)
dev = dev.remove_columns(["Unnamed: 0", "SpeakerID", "starttime", "duration", "original", "path", "DocID", "role", "Location", "detailed", "simple", "MurrealueOlli"])
dev = dev.cast_column("path_data", Audio(sampling_rate=16000))

skn_test['path_data'] = 'data/' + skn_test['path'].astype(str)
test = Dataset.from_pandas(skn_test)
test = test.remove_columns(["Unnamed: 0", "SpeakerID", "starttime", "duration", "original", "path", "DocID", "role", "Location", "detailed", "simple", "MurrealueOlli"])
test = test.cast_column("path_data", Audio(sampling_rate=16000))

####
import numpy
import os
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model
import torchaudio

# Load model and feature extractor
model_name = "GetmanY1/wav2vec2-base-fi-150k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

if USE_VOICE_CONVERSION:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)

    def ref_generator(filestart):
        ref_paths = []
        for path, dir, files in os.walk("SKN/la_data"):
            for file in files:
                filepath = os.path.join(path, file)
                if file.startswith(filestart) and os.path.getsize(filepath) > 1000000:
                    ref_paths.append(filepath)
        return knn_vc.get_matching_set(ref_paths[:30])

    matching_sets = [
        ("vc1", ref_generator('hame_hat1')),
        ("vc2", ref_generator('sate_koke1')),
        ("vc3", ref_generator('pohp_yii1')),
        ("vc4", ref_generator('kare_luu1'))
    ]
else:
    matching_sets = None

if USE_AUGMENTATION:
    pitch_steps = [2, 4, -2, -4]
    pitch_transforms = [torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=s) for s in pitch_steps]
else:
    pitch_transforms = None

from tqdm.auto import tqdm

def extract_embeddings(
    split,
    layers='all',
    offset=1,
    output_dir=f'speaker_partitions/{FOLDER}',
    save_metadata=True,
    device=None,
    batch_size=8,
    use_fp16=True,
    apply_original=True,
    matching_sets=None,
    apply_aug=USE_AUGMENTATION
):
    # Select original data and dataset by split
    if split == 'train':
        orig_data = skn_train
        dataset = train
    elif split == 'dev':
        orig_data = skn_dev
        dataset = dev
    else:
        orig_data = skn_test
        dataset = test

    layers_to_export = list(range(1, 13)) if layers == 'all' else sorted(set(layers))

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    length = len(orig_data)
    orig_data_reset = orig_data.reset_index(drop=True)

    variants = []
    if apply_original:
        variants.append('orig')
    if matching_sets:
        variants += [f"vc_{name}" for name, _ in matching_sets]
    if apply_aug and pitch_transforms:
        variants.append('aug_pitch')

    file_paths = {}
    for suffix in variants:
        variant_dir = os.path.join(output_dir, suffix)
        os.makedirs(variant_dir, exist_ok=True)
        file_paths[suffix] = {
            L: {
                "emb": f'{variant_dir}/{split}_embeddings_layer{L}.csv',
                "meta": f'{variant_dir}/{split}_embeddings_metadata_layer{L}.csv'
            }
            for L in layers_to_export
        }

    def count_csv_rows(path: str) -> int:
        """Return number of data rows in CSV written with one header row."""
        if not os.path.exists(path):
            return 0
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)
        return max(0, total_lines - 1)

    processed_counts = {suffix: {L: 0 for L in layers_to_export} for suffix in variants}
    for suffix in variants:
        for L in layers_to_export:
            emb_rows = count_csv_rows(file_paths[suffix][L]["emb"])
            if save_metadata:
                meta_rows = count_csv_rows(file_paths[suffix][L]["meta"])
                processed_counts[suffix][L] = min(emb_rows, meta_rows) if meta_rows > 0 else emb_rows
            else:
                processed_counts[suffix][L] = emb_rows

    variant_progress = {suffix: min(processed_counts[suffix].values()) for suffix in variants}

    num_batches_total = (length + batch_size - 1) // batch_size
    remaining_work_units = sum(max(0, length - variant_progress[s]) for s in variants)

    batch_pbar = tqdm(total=num_batches_total, desc=f"[{split}] Batches", unit="batch", leave=True)
    work_pbar = tqdm(total=remaining_work_units, desc=f"[{split}] Variants×Samples (remaining)", unit="item", leave=True)

    def process_variant(batch_arrays, suffix, batch_indices):
        """
        Writes only the portion of the batch that hasn't been processed yet
        """
        end_idx = batch_indices[-1]
        if all(processed_counts[suffix][L] > end_idx for L in layers_to_export):
            return  # Nothing to do

        x = feature_extractor(
            batch_arrays,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )
        input_values = x.input_values.to(device)

        with torch.no_grad():
            if use_fp16 and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    o = model(input_values, output_hidden_states=True)
            else:
                o = model(input_values, output_hidden_states=True)

        for L in layers_to_export:
            state = o.last_hidden_state if L == 12 else o.hidden_states[L + offset]
            means = torch.mean(state, dim=1)
            stds = torch.std(state, dim=1)

            layer_done_upto = processed_counts[suffix][L]
            keep_mask = [idx >= layer_done_upto for idx in batch_indices]

            if not any(keep_mask):
                continue

            batch_frames = []
            batch_meta_frames = []

            for j, keep in enumerate(keep_mask):
                if not keep:
                    continue
                df_means = pd.DataFrame(means[j].cpu().numpy().reshape(1, -1))
                df_stds = pd.DataFrame(stds[j].cpu().numpy().reshape(1, -1))
                df_layer = pd.concat([df_means, df_stds], axis=1)
                batch_frames.append(df_layer)

                if save_metadata:
                    meta_row = orig_data_reset.iloc[[batch_indices[j]]]
                    batch_meta_frames.append(
                        pd.concat([meta_row.reset_index(drop=True), df_layer], axis=1)
                    )

            df_full = pd.concat(batch_frames).reset_index(drop=True)
            emb_path = file_paths[suffix][L]["emb"]
            df_full.to_csv(
                emb_path,
                sep='\t',
                header=not os.path.exists(emb_path),
                index=False,
                mode='a'
            )

            if save_metadata:
                df_meta_full = pd.concat(batch_meta_frames).reset_index(drop=True)
                meta_path = file_paths[suffix][L]["meta"]
                df_meta_full.to_csv(
                    meta_path,
                    sep='\t',
                    header=not os.path.exists(meta_path),
                    index=False,
                    mode='a'
                )

            written_n = len(df_full)
            processed_counts[suffix][L] += written_n

        old_min = variant_progress[suffix]
        new_min = min(processed_counts[suffix].values())
        delta = max(0, new_min - old_min)
        if delta > 0:
            variant_progress[suffix] = new_min
            work_pbar.update(delta)
            work_pbar.set_postfix_str(f"variant={suffix}, done={new_min}/{length}")

    # Start from the earliest incomplete index across variants (min of variant_progress)
    global_start = min(variant_progress.values()) if variants else 0
    already_batches = global_start // batch_size
    if already_batches > 0:
        batch_pbar.update(already_batches)

    try:
        for start in range(global_start, length, batch_size):
            end = min(start + batch_size, length)
            raw_batch = [dataset[i]["path_data"]["array"] for i in range(start, end)]
            batch_indices = list(range(start, end))

            # Original
            if apply_original and 'orig' in variants:
                process_variant(raw_batch, suffix="orig", batch_indices=batch_indices)

            # Voice conversion sets
            if matching_sets:
                for name, vc_set in matching_sets:
                    suffix = f"vc_{name}"
                    if suffix not in variants:
                        continue
                    if all(processed_counts[suffix][L] >= end for L in layers_to_export):
                        continue
                    converted_batch = []
                    for array in raw_batch:
                        waveform = torch.tensor(array, dtype=torch.float).unsqueeze(0).to(device)
                        query_seq = knn_vc.get_features(waveform)
                        with torch.no_grad():
                            converted_waveform = knn_vc.match(query_seq, vc_set, topk=4)
                        converted_batch.append(converted_waveform.squeeze().cpu().numpy())
                    process_variant(converted_batch, suffix=suffix, batch_indices=batch_indices)

            # Augmentation
            if apply_aug and pitch_transforms and 'aug_pitch' in variants:
                if not all(processed_counts['aug_pitch'][L] >= end for L in layers_to_export):
                    transform = random.choice(pitch_transforms)
                    augmented_batch = []
                    for array in raw_batch:
                        waveform = torch.tensor(array, dtype=torch.float).unsqueeze(0)
                        noise = torch.randn_like(waveform) * 0.005
                        with torch.no_grad():
                            shifted_waveform = transform(waveform)
                            converted_waveform = shifted_waveform + noise
                        augmented_batch.append(converted_waveform.squeeze().numpy())
                    process_variant(augmented_batch, suffix="aug_pitch", batch_indices=batch_indices)

            batch_pbar.update(1)

    finally:
        batch_pbar.close()
        work_pbar.close()

extract_embeddings(
    split='train',
    layers=[1, 6, 12],
    batch_size=8,
    apply_original=True,
    matching_sets=matching_sets,
    apply_aug=USE_AUGMENTATION
)

extract_embeddings(
    split='dev',
    layers=[1, 6, 12],
    batch_size=8,
    apply_original=True,
    matching_sets=False,
    apply_aug=False
)

extract_embeddings(
    split='test',
    layers=[1, 6, 12],
    batch_size=8,
    apply_original=True,
    matching_sets=False,
    apply_aug=False
)
