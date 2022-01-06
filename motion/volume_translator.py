import tensorflow as tf
from network_configuration import TrackerConfiguration
from motion.recurrent import RecurrentLayer


class VolumeTranslator(tf.keras.layers.Layer):
    def __init__(self, configuration: TrackerConfiguration, **kwargs):
        super(VolumeTranslator, self).__init__(**kwargs)
        self.hidden_units = configuration.motion_hidden_units
        self.no_layers = configuration.motion_recurrent_layers

        self.no_features = configuration.no_features
        self.state_dense = tf.keras.layers.Dense(self.no_features)
        self.existence_dense = tf.keras.layers.Dense(1)
        self.state_recurrent = RecurrentLayer(configuration)
        self.existence_recurrent = RecurrentLayer(configuration, fan_in=1)
        self.no_particles = configuration.no_particles
        self.batch_size = configuration.batch_size

    def call(self, particle_states, existence, hidden_state):
        new_states, out_hidden_state = self.state_recurrent([particle_states, hidden_state])
        new_states = self.state_dense(new_states)

        new_existence, _ = self.existence_recurrent([existence, hidden_state])
        new_existence = self.existence_dense(new_existence)
        new_existence = tf.reshape(new_existence, [self.batch_size, self.no_particles])
        return new_states, new_existence, out_hidden_state
