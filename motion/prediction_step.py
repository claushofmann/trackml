import tensorflow as tf
from network_configuration import TrackerConfiguration
from motion.recurrent import RecurrentLayer


class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, configuration: TrackerConfiguration):
        super(PredictionLayer, self).__init__()
        self.hidden_units = configuration.motion_hidden_units
        self.no_particles = configuration.no_particles
        self.no_features = configuration.no_features
        self.batch_size = configuration.batch_size
        self.no_layers = configuration.motion_recurrent_layers
        self.recurrent = RecurrentLayer(configuration)
        self.transform_output = tf.keras.layers.Dense(self.no_features, activation=None, name='transf_output')

    def call(self, particle_states, hidden_state):
        """
        Predict the particles' state of the next detector layer

        :param particle_states: Particle states on the current detector layer
        :return: Particle states on the next detector layer
        """
        rnn_output, hidden_state = self.recurrent([particle_states, hidden_state])
        output = self.transform_output(rnn_output)
        return output, hidden_state
