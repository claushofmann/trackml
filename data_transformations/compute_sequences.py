import numpy as np
import pandas as pd
from volumes import *

def create_sequence_files(hits, cells, particles, truth, no_sequences, feature='theta'):
    merged = hits.merge(truth, on='hit_id')
    cutted, bins = pd.qcut(merged[feature], no_sequences, labels=[i for i in range(no_sequences)], retbins=True)
    if 'sequence_id' not in hits.columns:
        hits['sequence_id'] = cutted
    else:
        hits['sequence_id'] = hits['sequence_id'].astype(int) * no_sequences + cutted.astype(int)
    # truth['sequence_id'] = cutted

    return hits, cells, particles, truth

def make_sequences_file(name, hits, cells, particles, truth, no_sequences, feature='theta'):
    hits, cells, particles, truth = create_sequence_files(hits, cells, particles, truth, no_sequences, feature=feature)

    hits.to_csv(name + '-hits.csv', index=False)
    particles.to_csv(name + '-particles.csv', index=False)
    truth.to_csv(name + '-truth.csv', index=False)
    cells.to_csv(name + '-cells.csv', index=False)

if __name__ == '__main__':
    import load_data_luigi as ld
    event = ld.RootDetectorFiles(event_id=1000)
    event = ld.CreateAngles(create_from=event)
    event = ld.CreateNormalized(create_from=event)

    hits, cells, particles, truth = event.load()

    hits, cells, particles, truth = create_sequence_files( hits, cells, particles, truth, 14)
    hits, cells, particles, truth = create_sequence_files(hits, cells, particles, truth, 4, feature='phi')
