import tensorflow as tf
from motion.model_combiner_model_api import CombinedModelFactory
from association.model_factory import AssociationModelFactory
from motion.volume_translator import VolumeTranslator
from association.metrics import SparseAssociationLoss
from motion.metrics import ExistanceLoss, DistanceLoss
from network_configuration import TrackerConfiguration
from functools import reduce


class MultiVolumeModelFactory:
    def __init__(self, configuration: TrackerConfiguration):
        self.check_tree_volume_structure(configuration.root_volume, configuration.volume_dict)
        self.root_volume = configuration.root_volume
        self.volume_translation_dict = configuration.volume_dict
        self.no_particles = configuration.no_particles
        self.no_features = configuration.no_features
        self.batch_size = configuration.batch_size
        self.hidden_units = configuration.motion_hidden_units
        self.no_measurements = configuration.no_measurements
        self.type = configuration.type
        self.no_recurrent_layers = configuration.motion_recurrent_layers
        self.not_root_volumes = reduce(lambda a, b: a+b, self.volume_translation_dict.values())
        self.volumes = [self.root_volume] + self.not_root_volumes
        self.configuration = configuration
        self.assoc_model_factory = AssociationModelFactory(self.configuration)

    def check_tree_volume_structure(self, volume_root, volume_translation_dict):
        """
        Checks if the specified model structure is supported (each volume can right now only have one
        transition to another volume)
        :param volume_root:
        :param volume_translation_dict:
        :return:
        """
        # other volumes must transition from root
        assert volume_root in volume_translation_dict.keys()
        # assert volume_translation_dict[volume_root]

        # each volume can only have one direct predecessor
        volume_successors = reduce(lambda a, b: a+b, volume_translation_dict.values())
        assert len(volume_successors) == len(set(volume_successors))

        # nothing may transition to root
        assert volume_root not in volume_successors

    def create_model(self, single_between_assoc_model=True,
                     mock_assoc_model=False,
                     root_initializer=None,
                     loaded_volume_models=None,
                     loaded_between_assoc=None,
                     loaded_final_assoc=None,
                     train_motion_assoc_independently=False,
                     root_assoc=False):
        volume_measurements = {volume: tf.keras.Input([volume.value['no_layers'], self.no_measurements, self.no_features],
                                                      dtype=self.type,
                                                      name='measurements_{}'.format(volume))
                               for volume in self.volumes}
        volume_measurements_exist = {volume: tf.keras.Input([volume.value['no_layers'], self.no_measurements],
                                                                        dtype=bool,
                                                                        name='measurements_exist_{}'.format(volume))
                                     for volume in self.volumes}
        initial_particle_pos = tf.keras.Input([self.no_particles, self.no_features],
                                              name='inital_particles')
        initial_existence = tf.keras.Input([self.no_particles], name='initial_existence')

        true_assoc_matrix = {volume: tf.keras.Input([4], batch_size=None, dtype=tf.int64, name='true_assoc_{}'.format(volume))
                             for volume in self.volumes}

        volume_factories = {volume: CombinedModelFactory(configuration=self.configuration,
                                                         no_detector_layers=volume.value['no_layers'])
                            for volume in self.volumes}

        if single_between_assoc_model:
            if loaded_between_assoc is not None:
                between_assoc_model = loaded_between_assoc
            else:
                between_assoc_model = self.assoc_model_factory.create_model()
        else:
            between_assoc_model = None

        if loaded_volume_models:
            volume_models = {volume: loaded_volume_models[volume] for volume in self.not_root_volumes}
        else:
            volume_models = {volume: volume_factories[volume].new_model(between_assoc_model=between_assoc_model,
                                                                    hidden_state_input=True,
                                                                    hidden_state_output=True,
                                                                    mock_assoc_model=mock_assoc_model,
                                                                    train_motion_assoc_independently=train_motion_assoc_independently,
                                                                    strict_initializer=False)
                         for volume in self.not_root_volumes}

        if root_initializer is not None:
            volume_models[self.root_volume] = volume_factories[self.root_volume].new_model(between_assoc_model=between_assoc_model,
                                                                                           mock_assoc_model=mock_assoc_model,
                                                                                           initializer=root_initializer,
                                                                                           hidden_state_output=True,
                                                                                           train_motion_assoc_independently=train_motion_assoc_independently)
        elif not root_assoc:
            volume_models[self.root_volume] = volume_factories[self.root_volume].new_model(between_assoc_model=between_assoc_model,
                                                                                           hidden_state_output=True,
                                                                                           mock_assoc_model=mock_assoc_model,
                                                                                           train_motion_assoc_independently=train_motion_assoc_independently)
        else:
            volume_models[self.root_volume] = volume_factories[self.root_volume].new_model(mock_assoc_model=True,
                                                                                           hidden_state_output=True,
                                                                                           train_motion_assoc_independently=train_motion_assoc_independently)

        transition_models = {from_volume: {to_volume: VolumeTranslator(configuration=self.configuration)
                                           for to_volume in to_volumes}
                             for from_volume, to_volumes in self.volume_translation_dict.items()}

        pred_outputs = dict()
        upd_outputs = dict()
        inb_assoc_outputs = dict()
        existence_outputs = dict()

        def traverse_volume_tree(volume, hidden_state=None, initial_particle_pos=None, initial_existance=None, inject_assoc_model=False):
            inputs = [volume_measurements[volume], volume_measurements_exist[volume]]
            if initial_particle_pos is not None:
                inputs.append(initial_particle_pos)
                assert initial_existance is not None
                inputs.append(initial_existance)
            if inject_assoc_model:
                inputs.append(true_assoc_matrix[volume])
            if hidden_state is not None:
                inputs.append(hidden_state)

            print('Creating model for volume {}'.format(str(volume)))
            pred_particle_positions, upd_particle_positions, inb_assoc_matrix, existence_probabs, volume_hidden_state = volume_models[volume](inputs)

            pred_outputs[volume] = pred_particle_positions
            upd_outputs[volume] = upd_particle_positions
            inb_assoc_outputs[volume] = inb_assoc_matrix
            existence_outputs[volume] = existence_probabs
            for t_volume in self.volume_translation_dict[volume]:
                init_pos, transition_existance, trans_hidden_state = transition_models[volume][t_volume](upd_particle_positions[:,-1], existence_probabs[:,-1], volume_hidden_state)
                traverse_volume_tree(t_volume, trans_hidden_state, initial_particle_pos=init_pos, initial_existance=transition_existance, inject_assoc_model=mock_assoc_model)

        if root_initializer is None:
            traverse_volume_tree(self.root_volume, initial_particle_pos=initial_particle_pos, initial_existance=initial_existence, inject_assoc_model=mock_assoc_model or root_assoc)
        else:
            traverse_volume_tree(self.root_volume, inject_assoc_model=mock_assoc_model or root_assoc)

        inputs = list(volume_measurements.values()) + list(volume_measurements_exist.values())
        if root_initializer is None:
            inputs.append(initial_particle_pos)
            inputs.append(initial_existence)
        if mock_assoc_model:
            inputs = inputs + list(true_assoc_matrix.values())
        if root_assoc:
            inputs.append(true_assoc_matrix[self.root_volume])

        def map_outputs_to_list(output_dict, name):
            output_list = dict()
            for key, output in output_dict.items():
                full_name = '{}_{}'.format(name, key)
                output = tf.keras.layers.Lambda(lambda x: x, name=full_name)(output)
                output_list[full_name] = output
            return output_list

        pred_outputs_l = map_outputs_to_list(pred_outputs, 'predictions')
        upd_outputs_l = map_outputs_to_list(upd_outputs, 'updates')
        inb_assoc_outputs_l = map_outputs_to_list(inb_assoc_outputs, 'inb_assoc')
        existence_outputs_l = map_outputs_to_list(existence_outputs, 'exist')

        outputs = reduce(lambda a,b: {**a, **b}, [pred_outputs_l, upd_outputs_l, inb_assoc_outputs_l, existence_outputs_l])

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model

    def compile_model(self, model):
        loss_weights = {'predictions': lambda: 10.,
                        'updates': lambda: 10.,
                        'inb_assoc': lambda: 0.1,
                        'exist': lambda: 0.2}

        losses = {'predictions': lambda: DistanceLoss(n='pred'),
                  'updates': lambda: DistanceLoss(n='upd'),
                  'inb_assoc': SparseAssociationLoss,
                  'exist': lambda: (ExistanceLoss(weight_loss=1., weight_reg=1.) if self.configuration.existence_regularizazion
                  else ExistanceLoss(weight_loss=1., weight_reg=0.))}

        metrics = {
            'inb_assoc': lambda: None,
            'exist': tf.keras.metrics.BinaryAccuracy
        }

        def map_to_output_tensors(loss_configs):
            output_dict = {}
            for output_name, config in loss_configs.items():
                for volume in self.volumes:
                    output_dict['{}_{}'.format(output_name, volume)] = config()
            return output_dict

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            0.003, 20000, 0.95, staircase=True
        )

        model.compile(tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                      loss=map_to_output_tensors(losses),
                      loss_weights=map_to_output_tensors(loss_weights),
                      metrics=map_to_output_tensors(metrics))

        return model


if __name__ == '__main__':
    from volumes import *
    from network_configuration import row_col_config
    from collections import defaultdict
    volume_translation_dict = defaultdict(list)
    volume_translation_dict[Pixel.BARREL] = [Pixel.NEGATIVE_EC, Pixel.POSITIVE_EC, ShortStrip.BARREL]
    volume_translation_dict[ShortStrip.BARREL] = [ShortStrip.NEGATIVE_EC, ShortStrip.POSITIVE_EC]
    factory = MultiVolumeModelFactory(configuration=row_col_config)
    model = factory.create_model()
    factory.compile_model(model)
