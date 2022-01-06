import tensorflow as tf
from network_configuration import TrackerConfiguration


class AssociationLstm(tf.keras.layers.Layer):
    def __init__(self, configuration: TrackerConfiguration):
        super(AssociationLstm, self).__init__()
        self.hidden_units = configuration.assoc_lstm_hidden_units
        self.no_measurements = configuration.no_measurements
        self.no_particles = configuration.no_particles
        self.no_layers = 2
        self.hidden_nns = [tf.keras.layers.Dense(self.no_particles, activation='softmax') for _ in range(self.no_layers - 1)]
        self.final_nn = tf.keras.layers.Dense(self.no_particles, activation=None)
        self.nns = self.hidden_nns + [self.final_nn]
        self.hidden_lstms = [tf.keras.layers.LSTM(self.hidden_units, time_major=False, return_sequences=True) for _ in range(self.no_layers - 1)]
        self.final_lstm = tf.keras.layers.LSTM(self.hidden_units, time_major=False, return_sequences=True)
        self.lstms = self.hidden_lstms + [self.final_lstm]

    def call(self, c_matrix):
        #c_matrix_input = tf.reshape(c_matrix, [tf.shape(c_matrix)[0], -1])
        #c_matrix_input = tf.tile(tf.expand_dims(c_matrix_input, axis=1), [1, self.no_measurements, 1])
        c_matrix = tf.transpose(c_matrix, (0, 2, 1))
        lstm_output = c_matrix
        for lstm, nn in zip(self.lstms, self.nns):
            lstm_output = lstm(lstm_output)
            lstm_output = nn(lstm_output)
        # lstm_output = tf.nn.softmax(lstm_output, axis=2)
        lstm_output = tf.transpose(lstm_output, (0, 2, 1))
        return lstm_output
