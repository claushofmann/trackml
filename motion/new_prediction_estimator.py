import tensorflow as tf
from dataprepare import DataPreparer
from motion.motion_model import PredictionModel, UpdateModel
from motion.model_combiner_model_api import CombinedModelFactory
from network_configuration import TrackerConfiguration
from association.model_factory import AssociationModelFactory
import numpy as np
import os


class AbstractTracker:
    """
    Abstract class for a tracker for particle data.
    """
    def __init__(self, configuration:TrackerConfiguration):
        self.configuration = configuration
        self.no_sequences = configuration.bucket_parameter
        self.no_particles = configuration.no_particles
        self.dims = list(configuration.target_dimensions)
        self.no_features = len(self.dims)
        self.batch_size = configuration.batch_size
        self.hidden_units = configuration.motion_hidden_units
        self.no_measurements = configuration.no_measurements
        self.type = configuration.type
        self.no_recurrent_layers = configuration.motion_recurrent_layers
        self.association_model_factory = AssociationModelFactory(configuration=self.configuration,)

    def _create_model(self):
        """
        Create the TensorFlow graph with the optimizer for minimizing the target value

        :param minimize_update_loss: Include the loss of the update model in the term to be minimized
        """
        pass

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

    def get_model_path(self, model_name, predict_only=False):
        if predict_only:
            return os.path.join('trained_models', model_name + self.configuration.get_prediction_only_config_id() + '.tf')
        else:
            return os.path.join('trained_models', model_name + self.configuration.get_config_id() + '.tf')

    def map_volume_sample_to_keras_input(self, tuple):
        pass

    def map_generator_to_keras_input(self, generator):
        pass

    def save_model(self):
        pass


class AbstractSingleVolumeTracker(AbstractTracker):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.volume = configuration.root_volume
        self.no_layers = self.volume.value['no_layers']

        self._create_model()

    def map_generator_to_keras_input(self, generator):
        for d in generator:
            yield self.map_volume_sample_to_keras_input(list(d.values())[0])

    def train_on_volume(self, train_events, val_events):
        sampler = DataPreparer(self.no_particles, self.no_measurements, self.dims, self.batch_size, flatten_layer_dim=False, empty_particle_strategy='nan')
        train_generator = sampler.batch_generator(train_events, samples_per_event=self.batch_size, volume=self.volume, no_sequences=self.no_sequences)
        train_epoch_l = sampler.get_number_of_batches(train_events, self.batch_size, volume=self.volume)

        val_generator = sampler.batch_generator(val_events, self.batch_size, volume=self.volume, no_sequences=self.no_sequences)
        val_l = sampler.get_number_of_batches(val_events, self.batch_size, volume=self.volume)

        # self.model.run_eagerly = True

        self.model.fit(self.map_generator_to_keras_input(train_generator),
                       epochs=self.configuration.hm_epochs,
                       validation_data=self.map_generator_to_keras_input(val_generator),
                       steps_per_epoch=train_epoch_l,
                       validation_steps=val_l)

        self.save_model()

    def load_motion_model_weights(self):
        particle_placeholder = tf.zeros([self.batch_size, self.no_particles, self.no_features], dtype=self.type)
        measurement_placeholder = tf.zeros([self.batch_size, self.no_measurements, self.no_features], dtype=self.type)
        measurement_exist_placeholder = tf.zeros([self.batch_size, self.no_measurements], dtype=tf.bool)
        assoc_placeholder = tf.zeros([self.batch_size, self.no_particles, self.no_measurements], dtype=self.type)
        existance_placeholder = tf.zeros([self.batch_size, self.no_particles], dtype=self.type)
        hidden_state_placeholder = self.prediction_model.get_initial_state()
        self.prediction_model(particle_placeholder, hidden_state_placeholder)
        self.update_model(particle_placeholder, measurement_placeholder, measurement_exist_placeholder, assoc_placeholder,
                                         existance_placeholder, hidden_state_placeholder[1])
        self.prediction_model.load_weights(self.get_model_path('prediction'))
        self.update_model.load_weights(self.get_model_path('update'))


class MotionEstimator(AbstractSingleVolumeTracker):
    """
    Implementation of AbstractTracker which only trains the particle state prediction. It uses a mocked version
    of the association model
    """

    def map_volume_sample_to_keras_input(self, tuple):
        h, p, a, e = tuple
        init_parts = self.init_particles_remove_nan(p)
        h, h_e = self.init_measurements_remove_nan(h)
        keras_input = (h, h_e, init_parts, a), (p, p, a, e)
        return keras_input

    def _create_model(self):
        self.prediction_model = PredictionModel(self.no_particles, self.batch_size, self.hidden_units, self.no_measurements, self.no_features, no_layers=self.no_recurrent_layers)
        self.update_model = UpdateModel(self.no_particles, self.batch_size, self.hidden_units, self.no_measurements, self.no_features, no_layers=self.no_recurrent_layers)
        self.assoc_model2 = self.association_model_factory.create_model()
        model_factory = CombinedModelFactory(self.no_particles, self.batch_size, self.hidden_units, self.no_measurements, self.no_layers, self.no_features, self.type, no_recurrent_layers=self.no_recurrent_layers)
        self.model = model_factory.combine_models(self.prediction_model, self.update_model, association_model=None, final_association_model=self.assoc_model2)
        model_factory.compile_model(self.model)

    def save_model(self):
        self.prediction_model.save_weights(self.get_model_path('prediction'), save_format='tf')
        self.update_model.save_weights(self.get_model_path('update'), save_format='tf')
        self.assoc_model2.save_weights(self.get_model_path('assoc2'), save_format='tf')


class FullTracker(AbstractSingleVolumeTracker):
    """
    Implementation of AbstractTracker that implements full tracker capabilities using the true implementation of the
    association model
    """
    def __init__(self, configuration, train_independently=False, do_load=False):
        self.train_independently = train_independently
        self.do_load = do_load
        super().__init__(configuration)

    def get_assoc_model_path(self):
        # return os.path.join('..', 'association', get_assoc_model_path(self.configuration))
        return self.get_model_path('assoc2')

    def map_volume_sample_to_keras_input(self, tuple):
        h, p, a, e = tuple
        init_parts = self.init_particles_remove_nan(p)
        h, h_e = self.init_measurements_remove_nan(h)
        keras_input = (h, h_e, init_parts), (p, p, a, a, e)
        return keras_input

    def load_association_model_weights(self, model):
        particle_placeholder = tf.zeros([self.batch_size, self.no_particles, self.no_features], dtype=self.type)
        measurement_placeholder = tf.zeros([self.batch_size, self.no_measurements, self.no_features], dtype=self.type)
        model([particle_placeholder, measurement_placeholder])
        model.load_weights(self.get_assoc_model_path())

    def load_model_weights(self):
        self.model.load_weights(self.get_model_path('full'))

    def _create_model(self):
        self.prediction_model = PredictionModel(configuration=self.configuration)
        self.update_model = UpdateModel(configuration=self.configuration)

        self.association_model = self.association_model_factory.create_model()
        self.association_model2 = self.association_model_factory.create_model()

        model_factory = CombinedModelFactory(configuration=self.configuration, no_detector_layers=self.volume.value['no_layers'])
        self.model = model_factory.combine_models(self.prediction_model, self.update_model, association_model=self.association_model, final_association_model=self.association_model2, train_motion_assoc_independently=self.train_independently)
        if self.do_load:
            self.load_model_weights()
        model_factory.compile_model(self.model, train_motion_assoc_independently=self.train_independently)

    def save_model(self):
        self.model.save_weights(self.get_model_path('full'), save_format='tf')