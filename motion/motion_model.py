import tensorflow as tf
from motion.prediction_step import PredictionLayer
from motion.update_step import UpdateLayer, DenseMatrixProduct
from network_configuration import TrackerConfiguration


class PredictionModel(tf.keras.Model):
    def __init__(self, configuration: TrackerConfiguration):
        super(PredictionModel, self).__init__()
        self.no_particles = configuration.no_particles
        self.no_features = configuration.no_features
        self.batch_size = configuration.batch_size
        self.hidden_units = configuration.motion_hidden_units
        self.no_measurements = configuration.no_measurements
        self.no_recurrent_layers = configuration.motion_recurrent_layers
        self.prediction_layer = PredictionLayer(configuration)

    def call(self, particle_pos, in_hidden_state):
        predicted_particle_pos, out_hidden_state = self.prediction_layer(particle_pos, in_hidden_state)

        return predicted_particle_pos, out_hidden_state

    def get_initial_state(self):
        return self.prediction_layer.recurrent.get_initital_state()


class UpdateModel(tf.keras.Model):
    def __init__(self, configuration: TrackerConfiguration):
        super(UpdateModel, self).__init__()
        self.no_particles = configuration.no_particles
        self.no_features = configuration.no_features
        self.batch_size = configuration.batch_size
        self.hidden_units = configuration.motion_hidden_units
        self.no_measurements = configuration.no_measurements
        self.no_layers = configuration.motion_recurrent_layers
        self.update_layer = UpdateLayer(configuration)
        self.matrix_product_layer = DenseMatrixProduct(configuration)

    def call(self, inputs):
        predicted_particle_pos, measurements, measurements_exist, assoc_matrix, in_existance_probabs, hidden_state = inputs
        matrix_product = self.matrix_product_layer([predicted_particle_pos, measurements, measurements_exist, assoc_matrix, in_existance_probabs])
        updated_particle_pos, out_existance_probabs, hidden_state = self.update_layer(matrix_product, hidden_state, in_existance_probabs)

        # TODO Changed
        return updated_particle_pos, out_existance_probabs, hidden_state
