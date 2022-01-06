import tensorflow as tf
from network_configuration import TrackerConfiguration

class RecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, configuration: TrackerConfiguration, fan_in=None):
        super(RecurrentLayer, self).__init__()
        if configuration.motion_rnn == 'lstm':
            self.cells = [tf.keras.layers.LSTMCell(configuration.motion_hidden_units) for _ in range(configuration.motion_recurrent_layers)]
        elif configuration.motion_rnn == 'simple':
            self.cells = [tf.keras.layers.SimpleRNNCell(configuration.motion_hidden_units) for _ in range(configuration.motion_recurrent_layers)]
        self.configuration = configuration
        self.fan_in = fan_in if fan_in is not None else self.configuration.no_features

    def call(self, inputs):
        network_input, hidden_state = inputs
        network_input = tf.reshape(network_input, [self.configuration.batch_size * self.configuration.no_particles, self.fan_in],
                                   name='state_reshape')
        hidden_state = tf.reshape(hidden_state, self.get_state_shape())
        outputs = network_input
        out_hidden_states = []
        for i, recurrent_layer in enumerate(self.cells):
            state = hidden_state[i]
            outputs, out_state = recurrent_layer(outputs, state)
            out_hidden_states.append(out_state)
        hidden_state = tf.reshape(out_hidden_states, self.get_state_shape())
        outputs = tf.reshape(outputs, [self.configuration.batch_size, self.configuration.no_particles, self.configuration.motion_hidden_units])
        return outputs, hidden_state

    def get_initital_state(self):
        hidden_states = []
        for cell in self.cells:
            native_hidden = cell.get_initial_state(batch_size=self.configuration.batch_size*self.configuration.no_particles, dtype=self.dtype)
            hidden_states.append(native_hidden)
        return tf.reshape(hidden_states, self.get_state_shape())

    def get_state_shape(self):
        native_state_size = tf.convert_to_tensor(self.cells[0].get_initial_state(batch_size=self.configuration.batch_size * self.configuration.no_particles, dtype=self.configuration.type)).shape
        return [self.configuration.motion_recurrent_layers] + native_state_size