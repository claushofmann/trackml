import tensorflow as tf
from volumes import *
from collections import defaultdict

import json
import hashlib
import math
from functools import reduce

default_volume_dict = defaultdict(list)
default_volume_dict.update({
    Pixel.BARREL: [Pixel.POSITIVE_EC, Pixel.NEGATIVE_EC, ShortStrip.BARREL],
    ShortStrip.BARREL: [ShortStrip.POSITIVE_EC, ShortStrip.NEGATIVE_EC, LongStrip.BARREL],
    LongStrip.BARREL: [LongStrip.POSITIVE_EC, LongStrip.NEGATIVE_EC]
})

barrel_volume_dict = defaultdict(list)
barrel_volume_dict.update({
    Pixel.BARREL: [ShortStrip.BARREL],
    ShortStrip.BARREL: [LongStrip.BARREL]
})

pixel_volume_dict = defaultdict(list)
pixel_volume_dict.update({
    Pixel.BARREL: []
})

class TrackerConfiguration:
    """
    Contains configuration information for the Association Model training process
    """
    def __init__(self, total_no_particles=None, total_no_measurements=None,
                 batch_size=None, hm_epochs=None, dtype=None, dims_with_variances=dict(),
                 assoc_start_learning_rate=0.0003, assoc_decrease_steps=20000, assoc_decrease_percentage=0.95, assoc_model='lstm',
                 assoc_lstm_hidden_units=None, dense_layers=None, assoc_dense_row_col_no_layers=None, assoc_use_softmax_disctance=False,
                 assoc_use_projection=False, root_volume=None, motion_hidden_units=None, volume_dict=None, motion_recurrent_layers=1,
                 include_noise_hits=False, use_sparse_association=False, bucket_additional_capacity=0., samples_per_event=1, synthetic=False,
                 sequencing=None, bucket_parameter=None, motion_rnn='lstm', existence_regularization=False, existence_correction=False,
                 pixel_barrel_seeds=False):
        self.total_no_particles = total_no_particles
        self.total_no_measurements = total_no_measurements
        self.bucket_parameter=bucket_parameter
        self.sequencing = sequencing
        self.bucket_additional_capacity = bucket_additional_capacity
        if self.sequencing == 'annoy':
            self.no_particles = math.ceil((1 + bucket_additional_capacity) * (self.bucket_parameter / 7) * 10)
            self.no_measurements = math.ceil((1 + bucket_additional_capacity) * (self.bucket_parameter / 7) * 10)
        elif self.sequencing == 'theta' or self.sequencing == 't/p':
            self.no_particles = math.ceil((1 + bucket_additional_capacity) * (self.total_no_particles / self.bucket_parameter))
            self.no_measurements = math.ceil((1 + bucket_additional_capacity) * (self.total_no_measurements / self.bucket_parameter))
        else:
            self.no_particles = self.total_no_particles
            self.no_measurements = self.total_no_measurements
        self.assoc_lstm_hidden_units = assoc_lstm_hidden_units
        self.type = dtype
        self.batch_size = batch_size
        self.hm_epochs = hm_epochs
        # self.training_events = training_events
        # self.test_events = test_events
        self.assoc_start_learning_rate = assoc_start_learning_rate
        self.assoc_decrease_steps = assoc_decrease_steps
        self.assoc_decrease_percentage = assoc_decrease_percentage
        self.target_dimensions = list(dims_with_variances.keys())
        self.no_features = len(self.target_dimensions)
        self.dims_with_variances = dims_with_variances
        self.assoc_model = assoc_model
        self.assoc_dense_layers = tuple(dense_layers) if dense_layers else None
        self.assoc_dense_row_col_no_layers = assoc_dense_row_col_no_layers
        self.assoc_use_softmax_distance = assoc_use_softmax_disctance
        self.assoc_use_projection = assoc_use_projection
        # self.motion_no_prediction_layers = motion_no_prediction_layers
        self.root_volume = root_volume
        self.motion_hidden_units = motion_hidden_units
        self.volume_dict = volume_dict
        self.motion_recurrent_layers = motion_recurrent_layers
        self.include_noise_hits = include_noise_hits
        self.use_sparse_association = use_sparse_association
        self.volume_list = set([self.root_volume]) if self.volume_dict is None else set([self.root_volume] + reduce(lambda a, b: a + b, self.volume_dict.values()))
        self.samples_per_event = samples_per_event
        self.synthetic = synthetic
        self.motion_rnn = motion_rnn
        self.existence_regularizazion = existence_regularization
        self.existence_correction = existence_correction
        self.pixel_barrel_seeds = pixel_barrel_seeds

    def get_ser_dict(self):
        ser_dict = self.__dict__.copy()
        ser_dict['dtype'] = str(ser_dict['type'])
        del ser_dict['type']
        del ser_dict['samples_per_event']
        ser_dict['training_events'] = None
        ser_dict['test_events'] = None
        ser_dict['root_volume'] = None
        ser_dict['volume_dict'] = None
        # TODO Add serializibility for volume structure
        ser_dict['hm_epochs'] = None
        ser_dict['volume_list'] = None
        ser_dict['batch_size'] = None
        if ser_dict['motion_rnn'] == 'lstm':
            del ser_dict['motion_rnn']
        # ser_dict['assoc_start_learning_rate']=0.001
        if not ser_dict['synthetic']:
            del ser_dict['synthetic']
        if not ser_dict['pixel_barrel_seeds']:
            del ser_dict['pixel_barrel_seeds']
        return ser_dict

    def get_config_id(self):
        ser_dict = self.get_ser_dict()
        str_rep = json.dumps(ser_dict, sort_keys=True)
        return hashlib.sha256(str_rep.encode('utf-8')).hexdigest()

    def get_prediction_only_config_id(self):
        ser_dict = self.get_ser_dict()
        ser_dict['assoc_model'] = None
        ser_dict['assoc_dense_layers'] = None
        ser_dict['assoc_dense_row_col_no_layers'] = None
        ser_dict['assoc_use_softmax_distance'] = None
        ser_dict['assoc_use_projection'] = None
        ser_dict['assoc_start_learning_rate'] = None
        ser_dict['assoc_lstm_hidden_units'] = None
        ser_dict['assoc_decrease_steps'] = None
        ser_dict['assoc_decrease_percentage'] = None
        str_rep = json.dumps(ser_dict, sort_keys=True)
        return hashlib.sha256(str_rep.encode('utf-8')).hexdigest()


row_col_config = TrackerConfiguration(total_no_particles=12000,
                                      total_no_measurements=16000,
                                      batch_size=8,
                                      hm_epochs=2,
                                      dtype=tf.float32,
                                      dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                      assoc_start_learning_rate=0.003,
                                      assoc_decrease_steps=20000,
                                      assoc_decrease_percentage=0.95,
                                      assoc_model='row_col_dense',
                                      assoc_dense_row_col_no_layers=1,
                                      assoc_use_projection=False,
                                      assoc_use_softmax_disctance=False,
                                      motion_hidden_units=100,
                                      root_volume=Pixel.BARREL,
                                      volume_dict=default_volume_dict,
                                      motion_recurrent_layers=1,
                                      include_noise_hits=True,
                                      use_sparse_association=False,
                                      sequencing='t/p',
                                      bucket_parameter=56,
                                      bucket_additional_capacity=1.)

sampled_milan_config = TrackerConfiguration(total_no_particles=20,
                                            total_no_measurements=30,
                                            batch_size=12,
                                            hm_epochs=10,
                                            dtype=tf.float32,
                                            dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                            assoc_start_learning_rate=0.001,
                                            assoc_decrease_steps=20000,
                                            assoc_decrease_percentage=0.95,
                                            assoc_model='lstm',
                                            assoc_lstm_hidden_units=200,
                                            assoc_use_projection=False,
                                            assoc_use_softmax_disctance=False,
                                            motion_hidden_units=200,
                                            root_volume=Pixel.BARREL,
                                            volume_dict=default_volume_dict,
                                            motion_recurrent_layers=1,
                                            include_noise_hits=False,
                                            samples_per_event=12,
                                            existence_correction=True,
                                            existence_regularization=True,
                                            motion_rnn='simple')

sampled_milan_no_exist_regul_config = TrackerConfiguration(total_no_particles=20,
                                            total_no_measurements=30,
                                            batch_size=12,
                                            hm_epochs=10,
                                            dtype=tf.float32,
                                            dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                            assoc_start_learning_rate=0.001,
                                            assoc_decrease_steps=20000,
                                            assoc_decrease_percentage=0.95,
                                            assoc_model='lstm',
                                            assoc_lstm_hidden_units=200,
                                            assoc_use_projection=False,
                                            assoc_use_softmax_disctance=False,
                                            motion_hidden_units=200,
                                            root_volume=Pixel.BARREL,
                                            volume_dict=default_volume_dict,
                                            motion_recurrent_layers=1,
                                            include_noise_hits=False,
                                            samples_per_event=12,
                                            existence_correction=True,
                                            existence_regularization=False,
                                            motion_rnn='simple')

sampled_milan_lstm_motion_no_exist_regul_config = TrackerConfiguration(total_no_particles=20,
                                            total_no_measurements=30,
                                            batch_size=12,
                                            hm_epochs=10,
                                            dtype=tf.float32,
                                            dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                            assoc_start_learning_rate=0.001,
                                            assoc_decrease_steps=20000,
                                            assoc_decrease_percentage=0.95,
                                            assoc_model='lstm',
                                            assoc_lstm_hidden_units=200,
                                            assoc_use_projection=False,
                                            assoc_use_softmax_disctance=False,
                                            motion_hidden_units=200,
                                            root_volume=Pixel.BARREL,
                                            volume_dict=default_volume_dict,
                                            motion_recurrent_layers=1,
                                            include_noise_hits=False,
                                            samples_per_event=12,
                                            existence_correction=True,
                                            existence_regularization=False,
                                            motion_rnn='lstm')

sampled_dense_assoc_lstm_motion_config = TrackerConfiguration(total_no_particles=20,
                                            total_no_measurements=30,
                                            batch_size=12,
                                            hm_epochs=10,
                                            dtype=tf.float32,
                                            dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                            assoc_start_learning_rate=0.001,
                                            assoc_decrease_steps=20000,
                                            assoc_decrease_percentage=0.95,
                                            assoc_model='row_col_dense',
                                            assoc_dense_row_col_no_layers=1,
                                            assoc_use_projection=False,
                                            assoc_use_softmax_disctance=False,
                                            motion_hidden_units=200,
                                            root_volume=Pixel.BARREL,
                                            volume_dict=default_volume_dict,
                                            motion_recurrent_layers=1,
                                            include_noise_hits=False,
                                            samples_per_event=12,
                                            existence_correction=True,
                                            existence_regularization=False,
                                            motion_rnn='lstm')

sampled_dense_assoc_lstm_motion_no_exist_weighting_config = TrackerConfiguration(total_no_particles=20,
                                            total_no_measurements=30,
                                            batch_size=12,
                                            hm_epochs=10,
                                            dtype=tf.float32,
                                            dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                            assoc_start_learning_rate=0.001,
                                            assoc_decrease_steps=20000,
                                            assoc_decrease_percentage=0.95,
                                            assoc_model='row_col_dense',
                                            assoc_dense_row_col_no_layers=1,
                                            assoc_use_projection=False,
                                            assoc_use_softmax_disctance=False,
                                            motion_hidden_units=200,
                                            root_volume=Pixel.BARREL,
                                            volume_dict=default_volume_dict,
                                            motion_recurrent_layers=1,
                                            include_noise_hits=False,
                                            samples_per_event=12,
                                            existence_correction=False,
                                            existence_regularization=False,
                                            motion_rnn='lstm')

sampled_row_col_config_200 = TrackerConfiguration(total_no_particles=200,
                                              total_no_measurements=300,
                                              batch_size=12,
                                              hm_epochs=10,
                                              dtype=tf.float32,
                                              dims_with_variances={'x': 0.01, 'y': 0.01, 'z': 0.01, 'phi': 0.01, 'theta': 0.01},
                                              assoc_start_learning_rate=0.001,
                                              assoc_decrease_steps=20000,
                                              assoc_decrease_percentage=0.95,
                                              assoc_model='row_col_dense',
                                              assoc_dense_row_col_no_layers=1,
                                              assoc_use_projection=False,
                                              assoc_use_softmax_disctance=False,
                                              motion_hidden_units=200,
                                              root_volume=Pixel.BARREL,
                                              volume_dict=default_volume_dict,
                                              motion_recurrent_layers=1,
                                              include_noise_hits=False,
                                              samples_per_event=12,
                                              motion_rnn='lstm')
