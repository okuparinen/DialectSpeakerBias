import pandas as pd
import re

train_orig = pd.read_csv(f'speaker_partitions/split1/train.csv', sep='\t')
dev_orig = pd.read_csv(f'speaker_partitions/split1/validation.csv', sep='\t')
test_orig = pd.read_csv(f'speaker_partitions/split1/test.csv', sep='\t')
all_data = pd.concat([train_orig, dev_orig, test_orig], ignore_index=True)

alueet = list(set(all_data['named_dialect']))

for split in ['speaker_dependent']:
    meta_train = pd.read_csv(f'speaker_partitions/{split}/train.csv', sep='\t')
    meta_train = meta_train[meta_train['duration'] > 1.0]
    meta_train = meta_train[meta_train['duration'] < 20.0]
    meta_train = meta_train[:64100]
    meta_dev = pd.read_csv(f'speaker_partitions/{split}/validation.csv', sep='\t')
    meta_dev = meta_dev[meta_dev['duration'] > 1.0]
    meta_dev = meta_dev[meta_dev['duration'] < 20.0]
    meta_dev = meta_dev[:8180]
    meta_test = pd.read_csv(f'speaker_partitions/{split}/test.csv', sep='\t')
    meta_test = meta_test[meta_test['duration'] > 1.0]
    meta_test = meta_test[meta_test['duration'] < 20.0]
    meta_test = meta_test[:8180]
    for alue in alueet:
        speakers = list(set(all_data['DocID'][all_data['named_dialect'] == alue]))
        print(speakers)
        alue_train = meta_train[meta_train['DocID'].isin(speakers)]
        print(list(set(alue_train['DocID'])))
        alue_dev = meta_dev[meta_dev['DocID'].isin(speakers)]
        print(list(set(alue_dev['DocID'])))
        alue_test = meta_test[meta_test['DocID'].isin(speakers)]
        print(list(set(alue_test['DocID'])))
        alue_adjust = re.sub(' ', '_', alue)
        alue_train.to_csv(f'speaker_partitions/{split}/{alue_adjust}_train.csv', sep='\t', header=True, index=False)
        alue_dev.to_csv(f'speaker_partitions/{split}/{alue_adjust}_dev.csv', sep='\t', header=True, index=False)
        alue_test.to_csv(f'speaker_partitions/{split}/{alue_adjust}_test.csv', sep='\t', header=True, index=False)
