from annoy import AnnoyIndex
import numpy as np

# Code by Sabrina Amrouche. Source: https://github.com/greysab/ML-TR/blob/master/ML-TR.ipynb
def buildAnnoyIndex(data,metric="angular",ntrees=10):
    f = data.shape[1]
    t = AnnoyIndex(f,metric)
    for i,d in enumerate(data):
        t.add_item(i, d)
    t.build(ntrees) # more trees are slower to build but slightly more accurate
    return t

def make_sequences_annoy_file(name, hits, cells, particles, truth, no_per_bucket=20):
    merged = hits.merge(truth, on='hit_id')

    index = buildAnnoyIndex(merged[["x", "y", "z"]].values, metric="angular", ntrees=4)

    merged['sequence_id'] = -1

    i = 0
    while True:
        hits_without_bucket = merged[merged['sequence_id'] == -1]
        if len(hits_without_bucket) == 0:
            break
        n = np.random.choice(range(len(hits_without_bucket)))# choice of query position influcences bucket quality (sometime a lot)
        n_idx = hits_without_bucket.index[n]
        bucket_idx = np.array(index.get_nns_by_item(n_idx, no_per_bucket))
        # bucket = merged.iloc[bucket_idx]
        merged.loc[bucket_idx, 'sequence_id'] = i
        i = i + 1

    hits['sequence_id'] = merged['sequence_id']
    # truth['sequence_id'] = merged['sequence_id']

    hits.to_csv(name + '-hits.csv', index=False)
    particles.to_csv(name + '-particles.csv', index=False)
    truth.to_csv(name + '-truth.csv', index=False)
    cells.to_csv(name + '-cells.csv', index=False)

if __name__ == '__main__':
    import load_data_luigi as ld
    event = ld.RootDetectorFiles(event_id=1000)

    hits, cells, particles, truth = event.load()

    make_sequences_annoy_file('test', hits, cells, particles, truth, 400)
