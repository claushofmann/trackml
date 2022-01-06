from network_configuration import TrackerConfiguration
from dataprepare import DataPreparer
import numpy as np
import pandas as pd
import tensorflow as tf
from volumes import *
from trackml.score import score_event
from collections import defaultdict

output_d = defaultdict(list)

class OutputCreator:
    def __init__(self, configuration: TrackerConfiguration, model):
        self.model = model
        self.configuration = configuration
        self.sampler = DataPreparer(self.configuration, flatten_layer_dim=False, empty_particle_strategy='nan')

    def map_volume_sample_to_keras_input(self, d, output_d):
        input_d = dict()
        for key, t in d.items():
            h, h_id, p, a, a_w, e = t
            h, h_e = self.init_measurements_remove_nan(h)
            input_d['measurements_{}'.format(key)] = h
            input_d['measurements_exist_{}'.format(key)] = h_e
            input_d['true_assoc_{}'.format(key)] = a
            output_d['predictions_{}'.format(key)].append(p)
            output_d['true_assoc_{}'.format(key)].append(a)
            output_d['updates_{}'.format(key)].append(p)
            output_d['exist_{}'.format(key)].append(e)
        init_parts, particles_exist = self.init_particles_remove_nan(d[self.configuration.root_volume][2])
        input_d['inital_particles'] = init_parts
        input_d['initial_existence'] = particles_exist
        return input_d

    def get_true_and_predicted_output(self, event, return_sampled_truth=False):
        generator = self.sampler.batch_generator(event, samples_per_event=self.configuration.samples_per_event, volume_list=self.configuration.volume_list,
                                                 include_noise=self.configuration.include_noise_hits,
                                                 no_sequences=self.configuration.bucket_parameter, return_sampled_truth=return_sampled_truth)
        no_batches = self.sampler.get_number_of_batches(event, samples_per_event=self.configuration.samples_per_event, no_sequences=self.configuration.bucket_parameter)

        def extract_hit_ids(generator, hit_ids, sampled_truths, output_d):
            for d in generator:
                if return_sampled_truth:
                    d, sampled_truth = d
                    sampled_truths.append(sampled_truth)
                hit_ids.append({key: t[1] for key, t in d.items()})
                yield self.map_volume_sample_to_keras_input(d, output_d)

        hit_ids = []
        sampled_truths = []
        output_d = defaultdict(list)

        new_generator = extract_hit_ids(generator, hit_ids, sampled_truths, output_d)

        full_output = self.model.predict(new_generator, steps=no_batches)

        hit_ids = hit_ids[:no_batches]
        sampled_truths = sampled_truths[:no_batches]
        volume_hit_ids = {key: np.concatenate([d[key] for d in hit_ids], axis=0) for key in hit_ids[0]}

        for k, v in output_d.items():
            output_d[k] = np.concatenate(v[:no_batches], axis=0)

        if return_sampled_truth:
            return output_d, full_output, volume_hit_ids, sampled_truths
        else:
            return output_d, full_output, volume_hit_ids,


    def create_output(self, event, return_sampled_truth=False, existence_correction=False, ):
        tup = self.get_true_and_predicted_output(event, return_sampled_truth=return_sampled_truth)
        if return_sampled_truth:
            output_d, full_output, volume_hit_ids, sampled_truths = tup
        else:
            output_d, full_output, volume_hit_ids = tup

        assignments = []

        volume_associations = {key: full_output['inb_assoc_{}'.format(key)] for key in volume_hit_ids}
        volume_existances = {key: full_output['exist_{}'.format(key)] for key in volume_hit_ids}

        for key in volume_hit_ids:
            hit_ids = volume_hit_ids[key]
            associations = volume_associations[key]
            existances = volume_existances[key]

            # if self.configuration.samples_per_event > 1:
            #    hit_ids = hit_ids[:1]
            #    associations = associations[:1]
            #    existances = existances[:1]

            if existence_correction:
                associations = associations * np.reshape(tf.nn.sigmoid(existances), [-1, key.value['no_layers'], self.configuration.no_particles, 1])
            assigned_particles = np.argmax(associations, axis=2)
            assert assigned_particles.shape == (hit_ids.shape[0], key.value['no_layers'], self.configuration.no_measurements)
            for sequence_id, (batch_hit_ids, batch_assigned_particles) in enumerate(zip(hit_ids, assigned_particles)):
                relevant_hit_ids = batch_hit_ids[batch_hit_ids >= 0]
                relevant_assigned_particles = batch_assigned_particles[batch_hit_ids >= 0]
                particle_ids = np.full_like(relevant_assigned_particles, sequence_id * self.configuration.no_particles) + relevant_assigned_particles
                for hit_id, particle_id in zip(relevant_hit_ids, particle_ids):
                    assignments.append((hit_id, particle_id, sequence_id))

        assignments = pd.DataFrame(assignments, columns=['hit_id', 'track_id', 'sequence_id'])

        if return_sampled_truth:
            sampled_truths = sampled_truths[0]
            for i, s_t in enumerate(sampled_truths):
                if 'sequence_id' not in s_t.columns:
                    s_t.insert(0, 'sequence_id', i)
            sampled_truths = pd.concat(sampled_truths)
            return assignments, sampled_truths
        else:
            return assignments

    def score_output(self, event, score_on_full_event=True, score_sequences=False, existence_correction=False, score_only=None):

        class GeneratorWrapper:
            def __init__(self, event):
                self.event = event
            def load(self):
                yield event.load()
            def get_length(self):
                return 1

        if score_on_full_event:
            assignments = self.create_output(GeneratorWrapper(event), existence_correction=existence_correction)
            hits, _, _, truth = event.load()
            truth = hits.merge(truth, on='hit_id')
        else:
            assignments, truth = self.create_output(GeneratorWrapper(event), return_sampled_truth=True, existence_correction=existence_correction)

        if score_only:
            full_particles = truth[
                (truth['volume_id'] == Pixel.BARREL.value['id']) & (truth['layer_id'] == 2)]
            if score_only == 'full':
                truth = truth[truth['particle_id'].isin(full_particles['particle_id'])]
            elif score_only == 'detected':
                truth = truth[~truth['particle_id'].isin(full_particles['particle_id'])]
            assignments = assignments[assignments['hit_id'].isin(truth['hit_id'])]

        if score_sequences:
            scores = []
            for (s_id_1, sequence_truth), (s_id_2, sequence_assignments) in zip(truth.groupby('sequence_id'), assignments.groupby('sequence_id')):
                assert(s_id_1 == s_id_2)
                # find full particle tracks from truth, not only track parts contained in sequence
                # full_sequence_truth = truth[truth['particle_id'].isin(sequence_truth['particle_id'])]
                scores.append(score_event(sequence_truth, sequence_assignments))
            return scores
        else:
            score = score_event(truth, assignments)
            return score


    def init_particles_remove_nan(self, true_particle_states):
        init_parts = true_particle_states[:,0]
        particles_exist = ~np.isnan(init_parts)
        init_parts = np.where(~particles_exist, np.random.uniform(-0.5, 0.5, init_parts.shape), init_parts)
        particles_exist = np.any(particles_exist, axis=2)
        return init_parts, particles_exist

    def init_measurements_remove_nan(self, measurements):
        measurements_exist = ~np.any(np.isnan(measurements), axis=-1)
        measurements = np.where(np.expand_dims(measurements_exist, axis=-1), measurements, 0.)
        return measurements, measurements_exist

if __name__ == '__main__':
    from motion.multi_volume_tracker import PredictionMultiVolumeTracker, FullMultiVolumeTracker
    from network_configuration import *
    import load_data_luigi as ld

    config_to_use = sampled_row_col_config_200

    # This reduces the total amount of VRAM needed on the GPU drastically
    devices = tf.config.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    tracker = FullMultiVolumeTracker(config_to_use, do_load_full=True)
    # tracker = PredictionMultiVolumeTracker(config_to_use, do_load=True)
    output_creator = OutputCreator(config_to_use, tracker.model)

    train_data = ld.RootRangeDetectorFiles(start_range=1000, end_range=1001)
    if config_to_use.synthetic:
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.create_synthetic(7000))
    if config_to_use.sequencing == 'annoy':
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data,
                                                  derive_task=ld.create_annoy_sequences(config_to_use.bucket_parameter))
    train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.CreateAngles)
    if config_to_use.sequencing == 'theta':
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data,
                                                  derive_task=ld.create_sequences(config_to_use.bucket_parameter))
    if config_to_use.sequencing == 't/p':
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.create_sequences(14))
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data,
                                                  derive_task=ld.create_phi_sequences(4))
    train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.CreateNormalized)

    print(output_creator.score_output(train_data.requires()[0], score_on_full_event=False, score_sequences=True))
    print(output_creator.score_output(train_data.requires()[0], score_on_full_event=False, score_sequences=True, existence_correction=True))

