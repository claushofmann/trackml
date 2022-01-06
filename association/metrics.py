import tensorflow as tf


class AssociationLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        kwargs['name'] = 'association_loss'
        super(AssociationLoss, self).__init__(**kwargs)

    def call(self, true_matrix, predicted_matrix):
        true_association_scores = tf.where(tf.equal(true_matrix, 1), -tf.math.log(predicted_matrix + 0.0001), tf.zeros_like(predicted_matrix))
        no_assignments = tf.cast(tf.reduce_sum(true_matrix, axis=None), dtype=predicted_matrix.dtype)
        no_assignments = tf.where(no_assignments == 0., 1., no_assignments)
        loss = tf.reduce_sum(true_association_scores, axis=None) / no_assignments
        return loss

class SparseAssociationLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        kwargs['name'] = 'association_loss'
        super(SparseAssociationLoss, self).__init__(**kwargs)

    def call(self, true_matrix, predicted_matrix):
        true_matrix = tf.cast(true_matrix, tf.int64)
        true_association_scores = tf.gather_nd(-tf.math.log(predicted_matrix + 0.0001), true_matrix)
        no_assignments = tf.cast(tf.shape(true_matrix)[0], dtype=predicted_matrix.dtype)
        no_assignments = tf.where(no_assignments == 0., 1., no_assignments)
        loss = tf.reduce_sum(true_association_scores, axis=None) / no_assignments
        return loss

class SparseAssociationLossWeightable(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        kwargs['name'] = 'association_loss'
        super(SparseAssociationLossWeightable, self).__init__(**kwargs)

    def call(self, true_matrix, predicted_matrix):
        true_matrix, weights = true_matrix[:,:-1], true_matrix[:, -1]
        true_matrix = tf.cast(true_matrix, tf.int64)
        true_association_scores = tf.gather_nd(-tf.math.log(predicted_matrix + 0.0001), true_matrix)
        true_association_scores_weighted = true_association_scores * weights
        loss = tf.reduce_sum(true_association_scores_weighted, axis=None)
        return loss


class AbstractAssociationAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(AbstractAssociationAccuracy, self).__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(
            'total_correct', initializer='zeros', dtype=tf.int32)
        self.total = self.add_weight(
            'total', initializer='zeros', dtype=tf.int32)

    def result(self):
        return self.total_correct / self.total


class AssociationAccuracy(AbstractAssociationAccuracy):

    def update_state(self, true_matrix, predicted_matrix, sample_weight=None):
        true_matrix = tf.cast(true_matrix, tf.bool)
        has_particle = tf.math.reduce_any(true_matrix, axis=-2)
        association_max_vals = tf.math.reduce_max(predicted_matrix, axis=-2)
        true_assoc_scores = tf.where(tf.equal(true_matrix, True), predicted_matrix, tf.zeros_like(predicted_matrix))
        max_true_scores = tf.math.reduce_max(true_assoc_scores, axis=-2)
        correct_results = tf.math.logical_and(association_max_vals == max_true_scores, has_particle)
        self.total_correct.assign_add(tf.reduce_sum(tf.cast(correct_results, tf.int32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(true_matrix, tf.int32)))


class SparseAssociationAccuracy(AbstractAssociationAccuracy):

    def update_state(self, true_matrix, predicted_matrix, sample_weight=None):
        true_matrix = tf.cast(true_matrix, tf.int64)
        true_matrix = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(true_matrix, tf.ones(tf.shape(true_matrix)[0]),
                                                   dense_shape=predicted_matrix.shape)))
        true_matrix = tf.cast(true_matrix, tf.bool)
        has_particle = tf.math.reduce_any(true_matrix, axis=-2)
        association_max_vals = tf.math.reduce_max(predicted_matrix, axis=-2)
        true_assoc_scores = tf.where(tf.equal(true_matrix, True), predicted_matrix, tf.zeros_like(predicted_matrix))
        max_true_scores = tf.math.reduce_max(true_assoc_scores, axis=-2)
        correct_results = tf.math.logical_and(association_max_vals == max_true_scores, has_particle)
        self.total_correct.assign_add(tf.reduce_sum(tf.cast(correct_results, tf.int32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(true_matrix, tf.int32)))
