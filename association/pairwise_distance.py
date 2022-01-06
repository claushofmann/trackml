import tensorflow as tf
from math import nan


class NormalizeBatch(tf.keras.layers.Layer):
    def call(self, xs):
        count = tf.reduce_sum(tf.where(tf.math.is_nan(xs), tf.zeros_like(xs), tf.ones_like(xs)), axis=[1, 2])
        s = tf.reduce_sum(tf.where(tf.math.is_nan(xs), tf.zeros_like(xs), xs), axis=[1, 2])
        mean = s / count

        diffs = tf.reduce_sum(tf.where(tf.math.is_nan(xs), tf.zeros_like(xs), tf.square(xs)), axis=[1, 2])
        var = diffs / count - tf.square(mean)

        mean = tf.reshape(mean, shape=[-1, 1, 1])
        var = tf.reshape(var, shape=[-1, 1, 1])
        normalized = (xs - mean) / tf.sqrt(var)
        return normalized


class PairwiseDistance(tf.keras.layers.Layer):
    def call(self, measurements, predictions):
        batch_size = tf.shape(measurements)[0]
        no_predictions = predictions.shape[-2]
        no_measurements = measurements.shape[-2]
        dimensions = measurements.shape[-1]

        measurements_r = tf.reshape(measurements, (batch_size, 1, no_measurements, dimensions))
        predictions_r = tf.reshape(predictions, (batch_size, no_predictions, 1, dimensions))

        norm_matrix = tf.norm(measurements_r - predictions_r, 2, axis=-1)

        return norm_matrix