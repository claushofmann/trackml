import tensorflow as tf
from motion.metrics import DistanceLoss, ExistanceLoss
from association.metrics import AssociationLoss, AssociationAccuracy
from motion.motion_model import PredictionModel, UpdateModel
from association.model_factory import AssociationModelFactory
from network_configuration import TrackerConfiguration

class CombinedModel(tf.keras.models.Model):
    def __init__(self, configuration:TrackerConfiguration=None, no_detector_layers=None,
                 prediction_model: tf.keras.Model = None, update_model = None, association_model=None,
                 final_association_model=None,
                 initializer=None,
                 hidden_state_input=False,
                 hidden_state_output=False,
                 train_motion_assoc_independently=False,
                 strict_initializer=True):
        super(CombinedModel, self).__init__()
        self.no_particles = configuration.no_particles
        self.no_features = configuration.no_features
        self.no_detector_layers = no_detector_layers
        self.batch_size = configuration.batch_size
        self.hidden_units = configuration.motion_hidden_units
        self.no_measurements = configuration.no_measurements
        self.type = configuration.type
        self.configuration = configuration
        self.no_recurrent_layers = configuration.motion_recurrent_layers
        self.assoc_model_factory = AssociationModelFactory(self.configuration)
        self.assoc_model_factory = AssociationModelFactory(self.configuration)
        self.prediction_model = prediction_model
        self.update_model = update_model
        self.association_model = association_model
        self.initializer = initializer
        self.hidden_state_input = hidden_state_input
        self.hidden_state_output = hidden_state_output
        self.train_motion_assoc_independently = train_motion_assoc_independently
        self.strict_initializer = strict_initializer

    def call(self, inputs):

        the_model = self

        measurements, measurements_exist = inputs[:2]
        i = 2

        if self.initializer is None:
            initial_particle_pos = inputs[i]
            i = i + 1
            initial_existance = inputs[i]
            i = i + 1
        if self.association_model is None:
            true_assoc_list = inputs[i]
            i = i + 1
        if self.hidden_state_input:
            init_hidden_state = inputs[i]
            i = i + 1

        if self.hidden_state_input:
            hidden_state = init_hidden_state
        else:
            hidden_state = self.prediction_model.get_initial_state()

        if self.association_model is None:
            true_assoc_matrix = tf.scatter_nd(true_assoc_list, tf.ones(tf.shape(true_assoc_list)[0]), [self.batch_size, self.no_detector_layers, self.no_particles, self.no_measurements])

        if self.initializer is None:
            initial_position = initial_particle_pos
            init_exist = initial_existance
        else:
            initial_position, init_exist = self.initializer(measurements[0])
            # TODO Synchronize initial_position and association matrix!

        class LayerSlicer(tf.keras.layers.Layer):
            def call(self, x, layer):
                sliced = tf.sparse.slice(x, [0, layer, 0, 0], [the_model.batch_size, 1, the_model.no_particles, the_model.no_measurements])
                return tf.sparse.reshape(sliced, [the_model.batch_size, the_model.no_particles, the_model.no_measurements])

        if self.association_model is None:
            initial_assoc_matrix = true_assoc_matrix[:, 0]
        else:
            if self.train_motion_assoc_independently:
                initial_position_for_assoc = tf.stop_gradient(initial_position)
            else:
                initial_position_for_assoc = initial_position
            initial_assoc_matrix = self.association_model([initial_position_for_assoc, measurements[:, 0], measurements_exist[:, 0]])
        if self.train_motion_assoc_independently:
            initial_assoc_matrix_for_update = tf.stop_gradient(initial_assoc_matrix)
        else:
            initial_assoc_matrix_for_update = initial_assoc_matrix

        initial_update_position, initial_update_existance_probabs, _ = self.update_model([initial_position, measurements[:, 0],
                                                                                 measurements_exist[:, 0],
                                                                                 initial_assoc_matrix_for_update, init_exist,
                                                                                 hidden_state])

        if self.strict_initializer:
            current_position = initial_position
            current_existance_probabs = init_exist
        else:
            current_position = initial_update_position
            current_existance_probabs = initial_update_existance_probabs

        predicted_particle_positions = [initial_position]
        updated_particle_positions = [initial_update_position]
        in_between_assoc_matrix = [initial_assoc_matrix]
        existance_probabs = [current_existance_probabs]

        for i in range(1, self.no_detector_layers):
            l_measurements = measurements[:, i, ...]
            l_measurements_exist = measurements_exist[:, i, ...]
            current_position, hidden_state = self.prediction_model(current_position, hidden_state)
            predicted_particle_positions.append(current_position)

            if self.association_model is not None:
                if self.train_motion_assoc_independently:
                    current_position_for_association = tf.stop_gradient(current_position)
                else:
                    current_position_for_association = current_position
                current_association = self.association_model([current_position_for_association, l_measurements, l_measurements_exist])
            else:
                if self.configuration.use_sparse_association:
                    l_true_assoc_matrix = LayerSlicer()(true_assoc_matrix, i)
                else:
                    l_true_assoc_matrix = true_assoc_matrix[:, i]
                current_association = l_true_assoc_matrix

            if self.train_motion_assoc_independently:
                current_association_for_update = tf.stop_gradient(current_association)
            else:
                current_association_for_update = current_association

            current_position, current_existance_probabs, _ = self.update_model([current_position, l_measurements,
                                                                       l_measurements_exist, current_association_for_update,
                                                                       current_existance_probabs, hidden_state])

            updated_particle_positions.append(current_position)
            existance_probabs.append(current_existance_probabs)
            in_between_assoc_matrix.append(current_association)


        in_between_assoc_matrix = tf.stack(in_between_assoc_matrix, axis=1)

        predicted_particle_positions = tf.keras.layers.Lambda(lambda x: x, name='preditct')(
            tf.stack(predicted_particle_positions, axis=1))
        updated_particle_positions = tf.keras.layers.Lambda(lambda x: x, name='update')(
            tf.stack(updated_particle_positions, axis=1))
        in_between_assoc_matrix = tf.keras.layers.Lambda(lambda x: x, name='inb_association')(in_between_assoc_matrix)
        predicted_existance_probabs = tf.keras.layers.Lambda(lambda x: x, name='existance')(
            tf.stack(existance_probabs, axis=1))
        outputs = [predicted_particle_positions, updated_particle_positions, in_between_assoc_matrix, predicted_existance_probabs]
        if self.hidden_state_output:
            outputs.append(hidden_state)

        return outputs


class CombinedModelFactory:
    def __init__(self, configuration:TrackerConfiguration=None, no_detector_layers=None):
        self.no_particles = configuration.no_particles
        self.no_features = configuration.no_features
        self.no_detector_layers = no_detector_layers
        self.batch_size = configuration.batch_size
        self.hidden_units = configuration.motion_hidden_units
        self.no_measurements = configuration.no_measurements
        self.type = configuration.type
        self.configuration = configuration
        self.no_recurrent_layers = configuration.motion_recurrent_layers
        self.assoc_model_factory = AssociationModelFactory(self.configuration)

    def combine_models(self, prediction_model: tf.keras.Model, update_model, association_model=None,
                       final_association_model=None,
                       initializer=None,
                       hidden_state_input=False,
                       hidden_state_output=False,
                       train_motion_assoc_independently=False,
                       strict_initializer=True):

        return CombinedModel(self.configuration,
                             no_detector_layers=self.no_detector_layers,
                             prediction_model=prediction_model,
                             update_model=update_model,
                             association_model=association_model,
                             final_association_model=final_association_model,
                             initializer=initializer,
                             hidden_state_input=hidden_state_input,
                             hidden_state_output=hidden_state_output,
                             train_motion_assoc_independently=train_motion_assoc_independently,
                             strict_initializer=strict_initializer)


    def compile_model(self, model, train_motion_assoc_independently=False):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            0.003, 20000, 0.95, staircase=True
        )

        if train_motion_assoc_independently:
            losses = [DistanceLoss(), DistanceLoss(), AssociationLoss(), AssociationLoss(), ExistanceLoss()]
            loss_weights = [1., 1.5, 0.1, 0.1, 0.1]
        else:
            losses = [DistanceLoss(), DistanceLoss(), None, AssociationLoss(), ExistanceLoss()]
            loss_weights = [1., 1.5, 0., 0.1, 0.1]

        model.compile(tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                      loss=losses,
                      loss_weights=loss_weights,
                      metrics={'association': AssociationAccuracy(),
                               'existance': tf.keras.metrics.BinaryAccuracy()})

        return model

    def new_model(self, initializer=None, mock_assoc_model=False, between_assoc_model=None, final_assoc_model=None, hidden_state_input=False, hidden_state_output=False, train_motion_assoc_independently=False, strict_initializer=True):
        prediction_model = PredictionModel(configuration=self.configuration)
        update_model = UpdateModel(self.configuration)
        if mock_assoc_model:
            between_assoc_model = None
        else:
            if between_assoc_model is None:
                if self.configuration.use_sparse_association:
                    dense_assoc_model = self.assoc_model_factory.create_model()
                    between_assoc_model = SparseAssociation(dense_assoc_model, self.configuration)
                else:
                    between_assoc_model = self.assoc_model_factory.create_model()
        return self.combine_models(prediction_model, update_model, between_assoc_model, final_assoc_model, initializer, hidden_state_input, hidden_state_output, train_motion_assoc_independently, strict_initializer)
