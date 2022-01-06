from volumes import *
import numpy as np
import load_data_luigi as ld
from math import ceil
from collections import defaultdict
from network_configuration import TrackerConfiguration

class DataPreparer:
    def __init__(self, configuration:TrackerConfiguration, flatten_layer_dim=False, empty_particle_strategy='nan', sort_particle_inits_by=None):
        self.no_particles = configuration.no_particles
        self.no_measurements = configuration.no_measurements
        self.feature_dims = configuration.target_dimensions
        self.no_features = len(self.feature_dims)
        self.batch_size = configuration.batch_size
        self.flatten_layer_dim = flatten_layer_dim
        self.empty_particle_strategy = empty_particle_strategy
        self.sort_particle_inits_by = sort_particle_inits_by

    def add_particle_uncertainity(self, particle_df, std_dev_dict, inplace=True):
        if inplace:
            result_df = particle_df
        else:
            result_df = particle_df.copy()
        for index, row in particle_df.iterrows():
            for key, var_value in std_dev_dict.items():
                key = 't' + key
                result_df.at[index, key] = np.random.normal(loc=row[key], scale=var_value)
        return result_df

    def get_volume_arrays(self, volume_data, volume, part_id_lookup, particle_uncertainity_std_dev_dict=None):
        layer_list = []
        for layer_id in get_layer_ids(volume):
            layer_list.append(volume_data[volume_data['layer_id'] == layer_id])
        layer_hit_list = [l.reset_index() for l in layer_list]
        layer_hit_list = [l.sample(self.no_measurements).reset_index() if l.shape[0] > self.no_measurements else l for l
                          in layer_hit_list]

        particle_dim_names = ['t' + dim for dim in self.feature_dims]
        layer_particle_list = [l[particle_dim_names + ['particle_id']].groupby('particle_id').mean() for l in
                               layer_list]
        if particle_uncertainity_std_dev_dict is not None:
            for layer_particle_df in layer_particle_list:
                self.add_particle_uncertainity(layer_particle_df, particle_uncertainity_std_dev_dict)


        association_list = []
        association_weight_list = []
        # association_matrix = np.zeros([volume.value['no_layers'], self.no_particles, self.no_measurements],
        #                              dtype=np.bool)
        for layer_no, layer_hits in enumerate(layer_hit_list):
            for hit_idx in range(len(layer_hits)):
                particle_id = layer_hits.at[hit_idx, 'particle_id']
                if particle_id != 0:
                    particle_idx = part_id_lookup[particle_id]
                    # association_matrix[layer_no, particle_idx, hit_idx] = 1
                    association_list.append([layer_no, particle_idx, hit_idx])
                    association_weight_list.append(layer_hits.at[hit_idx, 'weight'])

        if self.empty_particle_strategy is 'nan':
            particle_array = np.full([volume.value['no_layers'], self.no_particles, self.no_features], np.nan)
        else:
            particle_array = np.random.uniform(-0.5, 0.5,
                                               [volume.value['no_layers'], self.no_particles, self.no_features])
        existances = np.zeros([volume.value['no_layers'], self.no_particles], dtype=np.bool)
        for layer_no, layer in enumerate(layer_particle_list):
            for particle_id, row in layer.iterrows():
                if particle_id != 0:
                    particle_idx = part_id_lookup[particle_id]
                    particle_array[layer_no][particle_idx] = row
                    existances[layer_no][particle_idx] = True

        # layer_hit_list = [np.append(np.array(layer[self.feature_dims]), np.random.uniform(-0.5, 0.5, [self.no_measurements - len(layer), self.no_features]), axis=0) for layer in layer_hit_list]
        layer_hit_id_list = [
            np.append(np.array(layer['hit_id']), np.full([self.no_measurements - len(layer)], -1), axis=0) for layer in
            layer_hit_list]
        layer_hit_list = [np.append(np.array(layer[self.feature_dims]),
                                    np.full([self.no_measurements - len(layer), self.no_features], np.nan), axis=0) for
                          layer in layer_hit_list]
        layer_permutations = [np.random.permutation(self.no_measurements) for _ in range(len(layer_hit_list))]
        permuted_hit_list = [layer[permutation, :] for layer, permutation in zip(layer_hit_list, layer_permutations)]
        permuted_hit_id_list = [layer[permutation] for layer, permutation in zip(layer_hit_id_list, layer_permutations)]

        hit_array = np.array(permuted_hit_list, dtype=np.float32)
        hit_id_array = np.array(permuted_hit_id_list, dtype=np.int64)

        inv_permutation = [np.argsort(perm) for perm in layer_permutations]

        permuted_association_list = [[layer_no, particle_idx, inv_permutation[layer_no][hit_idx]] for layer_no, particle_idx, hit_idx in association_list]
        #for layer_no, permutation in enumerate(layer_permutations):
        #    association_matrix[layer_no, ...] = association_matrix[layer_no, :, permutation].T

        return hit_array, hit_id_array, particle_array, np.reshape(permuted_association_list, [-1, 3]), association_weight_list, existances

    def single_sample(self, merged, volume=None, particle_uncertainity_std_dev_dict=None, volume_list=None, include_noise=False, sequence_no=None, return_sampled_truth=False):
        if volume_list is None:
            volume_list = [volume]
        if sequence_no is not None:
            merged = merged[merged['sequence_id'] == sequence_no]
        volume_merged = merged[merged['volume_id'].isin(volume.value['id'] for volume in volume_list)]

        possible_particle_ids = volume_merged['particle_id'].unique()
        possible_particle_ids = possible_particle_ids[possible_particle_ids != 0]
        if self.no_particles < possible_particle_ids.shape[0]:
            sampled_particle_ids = np.random.choice(possible_particle_ids, self.no_particles, replace=False)
        else:
            sampled_particle_ids = possible_particle_ids

        sampled_merged = volume_merged[volume_merged['particle_id'].isin(sampled_particle_ids)]
        sampled_merged = sampled_merged.rename(columns={'sequence_id_y': 'sequence_id'})
        if self.sort_particle_inits_by is None:
            sorted_particle_ids = sampled_merged.groupby('particle_id').min().sort_values(by=['volume_id', 'layer_id']).index
        else:
            sorted_particle_ids = sampled_merged.groupby('particle_id').min().sort_values(by=['volume_id', 'layer_id', self.sort_particle_inits_by]).index
        part_id_lookup = {particle_id: index for index, particle_id in enumerate(sorted_particle_ids)}

        if include_noise:
            sampled_merged = sampled_merged.append(volume_merged[volume_merged['particle_id'] == 0])

        volume_dict = dict()

        for volume in volume_list:

            volume_data = sampled_merged[sampled_merged['volume_id'] == volume.value['id']]

            hit_array, hit_id_array, particle_array, association_matrix, association_weights, existances = self.get_volume_arrays(volume_data, volume, part_id_lookup, particle_uncertainity_std_dev_dict)

            volume_dict[volume] = (hit_array, hit_id_array, particle_array, association_matrix, association_weights, existances)
        if return_sampled_truth:
            return volume_dict, sampled_merged
        else:
            return volume_dict

    def batch_generator(self, events, samples_per_event, volume=None, particle_uncertainity_dict=None, volume_list=None, include_noise=False, no_sequences=1, return_sampled_truth=False, randomize_sequences=False):
        def batch_instances(inst):
            def np_sparse_stack(indices, axis=0):
                return np.concatenate([np.concatenate(
                    [index[:, :axis], np.ones([index.shape[0], 1], dtype=np.int64) * i, index[:, axis:]],
                    axis=1) for i, index in enumerate(indices)], axis=0)
            return [
                        np.stack([a[0] for a in inst], axis=0), # hits
                        np.stack([a[1] for a in inst], axis=0), # hit_ids
                        np.stack([a[2] for a in inst], axis=0), #  particles
                        np_sparse_stack([a[3] for a in inst], axis=0), # association
                        np.concatenate([a[4] for a in inst]), # association weights
                        np.stack([a[5] for a in inst], axis=0) # existance
                    ]
        while True:
            i = 0
            batch = defaultdict(list)
            sampled_truths = []
            for instance in self.instance_generator(events, samples_per_event, particle_uncertainity_dict, volume=volume, volume_list=volume_list, include_noise=include_noise, no_sequences=no_sequences, return_sampled_truth=return_sampled_truth, randomize_sequences=randomize_sequences):
                i += 1
                if return_sampled_truth:
                    instance, sampled_truth = instance
                    sampled_truths.append(sampled_truth)
                for key, value in instance.items():
                    batch[key].append(value)
                if i == self.batch_size:
                    return_batch = {k: batch_instances(inst) for k, inst in batch.items()}
                    if return_sampled_truth:
                        yield return_batch, sampled_truths
                    else:
                        yield return_batch
                    batch = defaultdict(list)
                    sampled_truths = []
                    i = 0
            if i != 0:
                return_batch = {k: batch_instances(inst) for k,inst in batch.items()}
                if return_sampled_truth:
                    yield return_batch, sampled_truths
                else:
                    yield return_batch

    def instance_generator(self, events, samples_per_event, particle_uncertainity_dict, volume=None, volume_list=None, include_noise=False, no_sequences=1, return_sampled_truth=False, randomize_sequences=False):
        for event in events.load():
            hits, cells, particles, truth = event
            merged = hits.merge(truth, on='hit_id')
            for _ in range(samples_per_event):
                hits, _, _, _ = event
                if no_sequences is not None and no_sequences > 1:
                    sequences = hits['sequence_id'].unique()
                    if randomize_sequences:
                        sequences = np.random.choice(sequences, replace=False, size=no_sequences)
                else:
                    sequences = [0]
                for sequence in sequences:
                    if no_sequences is None or no_sequences == 1:
                        sample = self.single_sample(merged, volume, particle_uncertainity_dict, volume_list,
                                                    include_noise=include_noise, return_sampled_truth=return_sampled_truth)
                    else:
                        sample = self.single_sample(merged, volume, particle_uncertainity_dict, volume_list, include_noise=include_noise, sequence_no=sequence, return_sampled_truth=return_sampled_truth)
                    if return_sampled_truth:
                        sample, sampled_truth = sample
                    if self.flatten_layer_dim:
                        for volume in sample.values():
                            for t in zip(*volume):
                                yield {'instance': t}
                    else:
                        if return_sampled_truth:
                            yield sample, sampled_truth
                        else:
                            yield sample

    def get_number_of_batches(self, events: ld.RangeDetectorFiles, samples_per_event, volume=Pixel.BARREL, no_sequences=None):
        if no_sequences is None:
            no_sequences = 1
        if self.flatten_layer_dim:
            return ceil(events.get_length() * volume.value['no_layers'] * samples_per_event * no_sequences / self.batch_size)
        else:
            return ceil(events.get_length() * samples_per_event * no_sequences / self.batch_size)

if __name__ == '__main__':
    import load_data_luigi as ld
    root = ld.RootRangeDetectorFiles(start_range=1000, end_range=1005)
    phi = ld.DerivedRangeDetectorFiles(create_from=root, derive_task=ld.CreateAngles)
    normalized = ld.DerivedRangeDetectorFiles(create_from=phi, derive_task=ld.CreateNormalized)
    sampler = DataPreparer(200, 300, ['x', 'y', 'z'], batch_size=11, flatten_layer_dim=True)
    generator = sampler.batch_generator(normalized, 16)
    i = 0
    print(sampler.get_number_of_batches(normalized, samples_per_event=16))
    for t in generator:
        i += 1
        print(i)
    print('Okay')

