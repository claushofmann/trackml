import tensorflow as tf
from association.association_row_col import RowColumnDenseAssociation
from association.new_lstm import AssociationLstm
from association.pairwise_distance import PairwiseDistance, NormalizeBatch
from network_configuration import TrackerConfiguration
import numpy as np


def get_none_association(dtype=tf.float32):
    @tf.function
    def na(c_matrix):
        return tf.concat([c_matrix, tf.zeros([tf.shape(c_matrix)[0], tf.shape(c_matrix)[1], 1], dtype=dtype)], axis=2)
    return na


class AssociationModelFactory:
    def __init__(self, configuration:TrackerConfiguration):
        """

        :param configuration: Estimator Configuration object with conguration information
        """
        self.no_particles = configuration.no_particles
        self.no_measurements = configuration.no_measurements
        self.type = configuration.type
        self.batch_size = configuration.batch_size
        self.no_features = configuration.no_features
        self.assoc_model_type = configuration.assoc_model
        self.configuration = configuration

    def create_model(self):
        if self.assoc_model_type == 'lstm':
            association = AssociationLstm(configuration=self.configuration)
        elif self.assoc_model_type == 'row_col_dense':
            association = RowColumnDenseAssociation(configuration=self.configuration)
        elif self.assoc_model_type is None:
            association = tf.keras.layers.Lambda(get_none_association(self.type))
        else:
            raise Exception('{} is not a legal association model'.format(self.configuration.assoc_model))

        prediction_input = tf.keras.Input(shape=(self.no_particles, self.no_features), dtype=self.type)
        measurement_input = tf.keras.Input(shape=(self.no_measurements, self.no_features), dtype=self.type)
        measurement_exist_input = tf.keras.Input(shape=(self.no_measurements,), dtype=tf.bool)

        projected_prediction_input = prediction_input
        projected_measurement_input = measurement_input

        c_matrix_pd = PairwiseDistance()(projected_measurement_input, projected_prediction_input)

        if self.configuration.assoc_use_softmax_distance:
            @tf.function
            def f(x):
                # x = tf.where(tf.math.is_nan(x), -np.inf, -x)
                z = tf.nn.softmax(-x, axis=1)
                z = tf.where(tf.math.is_nan(z), tf.fill(tf.shape(z), 0.), z)
                return z
            c_matrix = tf.keras.layers.Lambda(f)(c_matrix_pd)
            c_matrix = tf.where(tf.reshape(measurement_exist_input, (self.batch_size, 1, self.no_measurements)),
                                c_matrix, -np.inf)
        else:
            # c_matrix = c_matrix_pd
            c_matrix = NormalizeBatch()(c_matrix_pd)
            # c_matrix = tf.nn.sigmoid(-c_matrix_pd)
            c_matrix = tf.where(tf.reshape(measurement_exist_input, (self.batch_size, 1, self.no_measurements)), c_matrix, tf.zeros_like(c_matrix))
        #c_matrix = NormalSimilarityLayer(len(self.target_dimensions))([measurement_input, prediction_input])
        #c_matrix = tf.where(tf.math.is_nan(c_matrix), tf.fill(tf.shape(c_matrix), 0.), c_matrix)

        output = association(c_matrix)
        # transp_output = tf.transpose(output, [0, 2, 1])
        output = tf.where(tf.reshape(measurement_exist_input, (self.batch_size, 1, self.no_measurements)), output, -np.inf)
        if self.assoc_model_type != 'hungarian':
            output = tf.nn.softmax(output, axis=1)
        output = tf.where(tf.reshape(measurement_exist_input, (self.batch_size, 1, self.no_measurements)), output, tf.zeros_like(output))
        #output = AttentionAssociation(len(self.target_dimensions), 2)([measurement_input, prediction_input])
        #output = tf.keras.layers.Lambda(get_none_association(self.dtype))(output)

        model = tf.keras.Model(inputs=[prediction_input, measurement_input, measurement_exist_input], outputs=[output])

        return model