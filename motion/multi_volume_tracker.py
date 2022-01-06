from motion.new_prediction_estimator import AbstractTracker
from dataprepare import DataPreparer
from motion.multi_volume_factory import MultiVolumeModelFactory
import tensorflow as tf
import sys

# Code to find if debugger is attached
trace = getattr(sys, 'gettrace', None)
debugger_attached = trace is not None and trace()
if debugger_attached:
    print('Debugger detected. Running Tensorflow in eager mode')


class AbstractMultiVolumeTracker(AbstractTracker):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.root_volume = configuration.root_volume
        self.volume_dict = configuration.volume_dict
        self.volume_list = configuration.volume_list

        self._create_model()
        self.model.run_eagerly = debugger_attached

    def get_full_model_path(self):
        pass

    def map_generator_to_keras_input(self, generator):
        for d in generator:
            yield self.map_volume_sample_to_keras_input(d)

    def train(self, train_events, val_events):
        sampler = DataPreparer(self.configuration,
                               flatten_layer_dim=False, empty_particle_strategy='nan', sort_particle_inits_by='phi')
        train_generator = sampler.batch_generator(train_events, samples_per_event=self.configuration.samples_per_event, volume_list=self.volume_list, include_noise=self.configuration.include_noise_hits, no_sequences=self.no_sequences, randomize_sequences=True)
        train_epoch_l = sampler.get_number_of_batches(train_events, samples_per_event=self.configuration.samples_per_event, no_sequences=self.no_sequences)

        val_generator = sampler.batch_generator(val_events, samples_per_event=self.configuration.samples_per_event, volume_list=self.volume_list, include_noise=self.configuration.include_noise_hits, no_sequences=self.no_sequences, randomize_sequences=True)
        val_l = sampler.get_number_of_batches(val_events, samples_per_event=self.configuration.samples_per_event, no_sequences=self.no_sequences)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.get_full_model_path(), save_best_only=False, save_weights_only=True, verbose=1, save_freq=100)

        self.model.fit(self.map_generator_to_keras_input(train_generator),
                       epochs=self.configuration.hm_epochs,
                       validation_data=self.map_generator_to_keras_input(val_generator),
                       steps_per_epoch=train_epoch_l,
                       validation_steps=val_l,
                       callbacks=[checkpoint])

        # self.save_model()


class PredictionMultiVolumeTracker(AbstractMultiVolumeTracker):
    def __init__(self, configuration, do_load=False):
        self.do_load = do_load
        super().__init__(configuration)

    def get_full_model_path(self):
        return self.get_model_path('multi_volume_predict', predict_only=True)

    def _create_model(self):
        factory = MultiVolumeModelFactory(self.configuration)
        self.model = factory.create_model(mock_assoc_model=True)
        if self.do_load:
            self.model.load_weights(self.get_full_model_path())
        factory.compile_model(self.model)

    def map_volume_sample_to_keras_input(self, d):
        input_d = dict()
        output_d = dict()
        for key, t in d.items():
            h, h_id, p, a, a_w, e = t
            h, h_e = self.init_measurements_remove_nan(h)
            input_d['measurements_{}'.format(key)] = h
            input_d['measurements_exist_{}'.format(key)] = h_e
            input_d['true_assoc_{}'.format(key)] = a
            output_d['predictions_{}'.format(key)] = p
            output_d['updates_{}'.format(key)] = p
            output_d['exist_{}'.format(key)] = e
        init_parts, particles_exist = self.init_particles_remove_nan(d[self.root_volume][2])
        init_parts = tf.convert_to_tensor(init_parts)
        particles_exist = tf.convert_to_tensor(particles_exist)
        input_d['inital_particles'] = init_parts
        input_d['initial_existence'] = particles_exist
        return input_d, output_d

    def save_model(self):
        self.model.save_weights(self.get_full_model_path(), save_format='tf')


class FullMultiVolumeTracker(AbstractMultiVolumeTracker):
    def __init__(self, configuration, do_load_full=False, do_load_predict=False):
        self.do_load_full = do_load_full
        self.do_load_predict = do_load_predict
        super().__init__(configuration)

    def get_full_model_path(self):
        return self.get_model_path('multi_volume')

    def _create_model(self):
        factory = MultiVolumeModelFactory(self.configuration)
        self.model = factory.create_model(train_motion_assoc_independently=True, single_between_assoc_model=False, root_assoc=False)
        if self.do_load_full:
            self.model.load_weights(self.get_model_path('multi_volume'))
        elif self.do_load_predict:
            model_path = self.get_model_path('multi_volume_predict', predict_only=True)
            print('Loading from model path {}'.format(model_path))
            self.model.load_weights(model_path)
        factory.compile_model(self.model)

    def map_volume_sample_to_keras_input(self, d):
        input_d = dict()
        output_d = dict()
        for key, t in d.items():
            h, h_id, p, a, a_w, e = t
            h, h_e = self.init_measurements_remove_nan(h)
            input_d['measurements_{}'.format(key)] = h
            input_d['measurements_exist_{}'.format(key)] = h_e
            output_d['predictions_{}'.format(key)] = p
            output_d['updates_{}'.format(key)] = p
            # output_d['inb_assoc_{}'.format(key)] = np.concatenate([a, np.expand_dims(a_w, axis=-1)], axis=-1)
            output_d['inb_assoc_{}'.format(key)] = a
            output_d['exist_{}'.format(key)] = e
        init_parts, particles_exist = self.init_particles_remove_nan(d[self.root_volume][2])
        input_d['inital_particles'] = init_parts
        input_d['initial_existence'] = particles_exist
        input_d['true_assoc_{}'.format(self.root_volume)] = d[self.root_volume][3]
        return input_d, output_d

    def load_pre_trained_weights(self):
        for model_name, volume_model in self.volume_models.items():
            volume_model.load_weights(self.get_model_path(str(model_name)))
        self.between_assoc.load_weights(self.get_model_path('final_assoc'))
        self.final_assoc.load_weights(self.get_model_path('final_assoc'))

    def save_model(self):
        self.model.save_weights(self.get_model_path('multi_volume'), save_format='tf')

