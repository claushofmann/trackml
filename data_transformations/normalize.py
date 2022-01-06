from volumes import *


def transformation(volume):
    def t(column):
        minimum, maximum = volume.value['geometry'][column.name[-1]]
        return (column - minimum) / (maximum - minimum) - 0.5
    return t


def inverse_transformation(volume):
    def t(column):
        minimum, maximum = volume.value['geometry'][column.name[-1]]
        return (column + 0.5) * (maximum - minimum) + minimum
    return t


def make_normalized_file(name, hits, cells, particles, truth):
    def normalize(data, transform_columns, keep_columns=None):
        if keep_columns is not None:
            transf_data = data[keep_columns].copy()
            transf_data[transform_columns] = data[transform_columns]
        else:
            transf_data = data.copy()
        for volume_id, d in data.groupby('volume_id'):
            volume = vol_id_index[volume_id]
            transf_data.loc[d.index, transform_columns] = d[transform_columns].transform(transformation(volume))
        return transf_data

    joined = hits.merge(truth, on='hit_id')

    transf_hits = normalize(joined, ['x', 'y', 'z'], keep_columns=hits.columns)
    transf_hits.to_csv(name + '-hits.csv', index=False)

    transf_truth = normalize(joined, ['tx', 'ty', 'tz'], keep_columns=truth.columns)
    transf_truth.to_csv(name + '-truth.csv', index=False)

    particles.to_csv(name + '-particles.csv', index=False)

    cells.to_csv(name + '-cells.csv', index=False)
