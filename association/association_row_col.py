import tensorflow as tf
from network_configuration import TrackerConfiguration


class RowColumnDenseAssociation(tf.keras.layers.Layer):
    def __init__(self, configuration:TrackerConfiguration):
        super(RowColumnDenseAssociation, self).__init__()
        self.no_layers = configuration.assoc_dense_row_col_no_layers
        self.no_measurements = configuration.no_measurements
        self.no_particles = configuration.no_particles
        self.first_layers1 = [tf.keras.layers.Dense(self.no_measurements) for _ in range(self.no_layers - 1)]
        self.first_layers2 = [tf.keras.layers.Dense(self.no_particles) for _ in range(self.no_layers - 1)]
        self.final_layer1 = tf.keras.layers.Dense(self.no_measurements)
        self.final_layer2 = tf.keras.layers.Dense(self.no_particles)
        self.layers1 = self.first_layers1 + [self.final_layer1]
        self.layers2 = self.first_layers2 + [self.final_layer2]

    def call(self, c_matrix):
        current = c_matrix
        for layer1, layer2 in zip(self.layers1, self.layers2):
            current = layer1(current)
            current = tf.transpose(current, (0,2,1))
            current = layer2(current)
            current = tf.transpose(current, (0,2,1))
        return current
