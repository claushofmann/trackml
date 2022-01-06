import tensorflow as tf
from network_configuration import TrackerConfiguration
from motion.recurrent import RecurrentLayer

class MatrixProduct(tf.keras.layers.Layer):
    def __init__(self, configuration:TrackerConfiguration):
        super(MatrixProduct, self).__init__()
        self.no_particles = configuration.no_particles
        self.no_measurements = configuration.no_measurements
        self.no_features = configuration.no_features
        self.prediction_size = self.no_particles * self.no_features
        self.batch_size = configuration.batch_size
        self.configuration = configuration

    def call(self, predicted_states, measurements, measurements_exist, assignment_probabilities, particles_existance):
        pass


class DenseMatrixProduct(MatrixProduct):
    def call(self, inputs):
        predicted_states, measurements, measurements_exist, assignment_probabilities, particles_existance = inputs
        measurement_sum = tf.reshape(tf.reduce_sum(assignment_probabilities, axis=2), [self.batch_size, self.no_particles, 1])
        measurement_sum = tf.where(measurement_sum == 0., 1., measurement_sum)
        assignment_probabilities_weighted = assignment_probabilities / measurement_sum
        if self.configuration.existence_correction:
            assignment_probabilities_weighted = assignment_probabilities_weighted * tf.reshape(
                tf.nn.sigmoid(particles_existance), [self.batch_size, self.no_particles, 1])

        updated_states = tf.einsum('ijk,ilj->ilk', measurements, assignment_probabilities_weighted)

        if self.configuration.existence_correction:
            updated_states = updated_states + tf.reshape((1. - tf.nn.sigmoid(particles_existance)),
                                                                                  [self.batch_size, self.no_particles,
                                                                                   1]) * predicted_states

        return updated_states

class ClassicMatrixProduct(MatrixProduct):
    def call(self, inputs):
        predicted_states, measurements, measurements_exist, assignment_probabilities, particles_existance = inputs

        measured_states_tiled = tf.tile(measurements, [1, 1, self.no_particles])
        measured_states_multip = tf.reshape(measured_states_tiled,
                                            [self.batch_size, self.no_measurements, self.no_particles,
                                             self.no_features])
        measured_states_transposed = tf.transpose(measured_states_multip, [0, 3, 1, 2])
        pred_states_for_concatination = tf.reshape(predicted_states,
                                                   [self.batch_size, 1, self.no_particles, self.no_features])
        pred_states_transposed = tf.transpose(pred_states_for_concatination, [0, 3, 1, 2])
        concat_input = tf.concat([measured_states_transposed, pred_states_transposed], 2)

        measurement_sum = tf.reshape(tf.reduce_sum(assignment_probabilities, axis=2),
                                     [self.batch_size, self.no_particles, 1])
        measurement_sum = tf.where(measurement_sum == 0., 1., measurement_sum)
        assignment_probabilities_weighted = assignment_probabilities / measurement_sum
        assignment_probabilities_existence_corrected = assignment_probabilities_weighted * tf.reshape(
            particles_existance, [self.batch_size, self.no_particles, 1])
        assignment_probabilities_existence_corrected = tf.concat([assignment_probabilities_existence_corrected,
                                                                  tf.reshape(1. - particles_existance,
                                                                             [self.batch_size, self.no_particles,
                                                                              1])], axis=2)
        assignment_probabilities_reshaped = tf.reshape(assignment_probabilities_existence_corrected,
                                                       [self.batch_size, 1, self.no_particles,
                                                        self.no_measurements + 1])
        assignment_probabilities_tiled = tf.tile(assignment_probabilities_reshaped, [1, self.no_features, 1, 1])

        # The matrix is multiplied in a very special way
        matrix_product = tf.einsum('afij,afji->afi', assignment_probabilities_tiled, concat_input)
        matrix_product = tf.transpose(matrix_product, [0, 2, 1])
        return matrix_product


class UpdateLayer(tf.keras.layers.Layer):

    def __init__(self, configuration: TrackerConfiguration):
        super(UpdateLayer, self).__init__()
        self.no_particles = configuration.no_particles
        self.no_measurements = configuration.no_measurements
        self.no_features = configuration.no_features
        self.prediction_size = self.no_particles * self.no_features
        self.hidden_units = configuration.motion_hidden_units

        self.batch_size = configuration.batch_size
        self.no_layers = configuration.motion_recurrent_layers
        self.recurrent = RecurrentLayer(configuration)
        # self.transform_matrix_product = tf.keras.layers.Dense(self.state_size, name='transf_matrix_product')

        # TODO Marker here
        self.transform_update_output = tf.keras.layers.Dense(self.no_features, name='transf_update_output')
        self.transform_to_existance_prob = tf.keras.layers.Dense(1, name='transf_to_exstance_prob') # no sigmoid, that is done in loss!

    def call(self, matrix_product, hidden_state, existance_probabilities):
        """
        Invoke the update model

        :param matrix_product:
        :param existance_probabilities: Probability of existance for every particle
        :param hidden_state: Hidden state from the state prediction
        :return: Tuple:
            - updated_state: Updated particle states
            - existance_probabilities: Probabilities of existance for the particles
        """

        matrix_product_transformed, hidden_state = self.recurrent([matrix_product, hidden_state])

        updated_state = self.transform_update_output(matrix_product_transformed)

        matrix_product_for_existance = tf.reshape(matrix_product_transformed, [self.batch_size, self.no_particles, -1])
        existance_probabilities = tf.reshape(self.transform_to_existance_prob(matrix_product_for_existance), [self.batch_size, self.no_particles])

        return updated_state, existance_probabilities, hidden_state
