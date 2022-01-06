import tensorflow as tf


class DistanceLoss(tf.keras.losses.Loss):
    def __init__(self, n='dist'):
        super(DistanceLoss, self).__init__(name='distance_loss')
        self.n = n

    def call(self, true_particle_state, predicted_particle_state):
        """
        Get the distance between the true particle state and the predicted particle state for the loss of the
        motion model

        :param true_particle_state: The true state of the particles
        :param predicted_particle_state: The state of the particle predicted by the prediction or update model
        :return: The mean distance between predicted and true states
        """
        particles_exist = ~tf.reduce_any(tf.math.is_nan(true_particle_state), axis=-1)
        true_particle_state = tf.where(tf.expand_dims(particles_exist, axis=-1), true_particle_state, tf.zeros_like(true_particle_state))
        loss = tf.keras.losses.MSE(true_particle_state, predicted_particle_state)
        loss = tf.where(particles_exist, loss, tf.zeros_like(loss))
        return loss

class DistanceWeightableLoss(tf.keras.losses.Loss):
    def __init__(self, n='dist'):
        super(DistanceWeightableLoss, self).__init__(name='distance_loss')
        self.n = n

    def call(self, true_particle_state, predicted_particle_state):
        """
        Get the distance between the true particle state and the predicted particle state for the loss of the
        motion model

        :param true_particle_state: The true state of the particles
        :param predicted_particle_state: The state of the particle predicted by the prediction or update model
        :return: The mean distance between predicted and true states
        """
        particles_exist = ~tf.reduce_any(tf.math.is_nan(true_particle_state), axis=-1)
        difference = predicted_particle_state - true_particle_state
        difference = tf.where(tf.expand_dims(particles_exist, axis=-1), difference, tf.zeros_like(difference))
        loss = tf.norm(difference, axis=-1)
        per_particle_loss = tf.math.reduce_sum(loss, axis=1)
        return per_particle_loss


class ExistanceLoss(tf.keras.losses.Loss):
    def __init__(self, weight_loss, weight_reg):
        super(ExistanceLoss, self).__init__(name='existance_loss')
        self.weight_loss = weight_loss
        self.weight_reg = weight_reg

    def call(self, true_existance_probabs, predicted_existance_probabs):
        cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(true_existance_probabs, tf.float32), logits=predicted_existance_probabs)
        probabs_without_first = tf.nn.sigmoid(predicted_existance_probabs[:, 1:])
        probabs_without_last = tf.nn.sigmoid(predicted_existance_probabs[:, :-1])
        regularization_loss = tf.reduce_mean(tf.abs(probabs_without_first - probabs_without_last))
        total_loss = self.weight_loss * cross_entropy_loss + self.weight_reg * regularization_loss
        return total_loss