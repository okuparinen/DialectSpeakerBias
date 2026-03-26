import pandas as pd
import numpy as np
import re

train_orig = pd.read_csv(f'speaker_partitions/split1/train.csv', sep='\t')
dev_orig = pd.read_csv(f'speaker_partitions/split1/validation.csv', sep='\t')
test_orig = pd.read_csv(f'speaker_partitions/split1/test.csv', sep='\t')
all_data = pd.concat([train_orig, dev_orig, test_orig], ignore_index=True)

alueet = list(set(all_data['MurrealueOlli']))
numbers = list(range(1,9))
alue_dict = dict(zip(alueet, numbers))

for split in ['split1']:
    for layer in [1, 6, 12]:
        dialect_frames = []
        for alue in alueet:
            speakers = list(set(all_data['DocID'][all_data['MurrealueOlli'] == alue]))
            method = f'vc_vc{alue_dict[alue]}'
            train_per_method = pd.read_csv(f'speaker_partitions/{split}/{method}/train_embeddings_metadata_layer{layer}.csv', sep='\t')
            dialect_set = train_per_method[train_per_method['DocID'].isin(speakers)]
            dialect_frames.append(dialect_set)
        dialect_df = pd.concat(dialect_frames, ignore_index=True)
        dialect_df.to_csv(f'speaker_partitions/{split}/dialect-per-speaker/train_embeddings_metadata_layer{layer}.csv', sep='\t', header=True, index=False)
