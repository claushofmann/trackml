import sys
sys.path.append('../')
import tensorflow as tf

from motion.multi_volume_tracker import PredictionMultiVolumeTracker
import load_data_luigi as ld
from volumes import *
from network_configuration import *


if __name__ == '__main__':

    config_to_use = sampled_dense_assoc_lstm_motion_config

    # This reduces the total amount of VRAM needed on the GPU drastically
    # devices = tf.config.list_physical_devices('GPU')
    # for device in devices:
    #    tf.config.experimental.set_memory_growth(device, True)

    train_data = ld.RootRangeDetectorFiles(start_range=1000, end_range=1400)
    if config_to_use.synthetic:
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.create_synthetic(7000))
    if config_to_use.sequencing == 'annoy':
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data,
                                                  derive_task=ld.create_annoy_sequences(config_to_use.bucket_parameter))
    train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.CreateAngles)
    if config_to_use.sequencing == 'theta':
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data,
                                                  derive_task=ld.create_sequences(config_to_use.bucket_parameter))
    if config_to_use.sequencing == 't/p':
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.create_sequences(14))
        train_data = ld.DerivedRangeDetectorFiles(create_from=train_data,
                                                  derive_task=ld.create_phi_sequences(4))
    train_data = ld.DerivedRangeDetectorFiles(create_from=train_data, derive_task=ld.CreateNormalized)

    test_data = ld.RootRangeDetectorFiles(start_range=1400, end_range=1440)
    if config_to_use.synthetic:
        test_data = ld.DerivedRangeDetectorFiles(create_from=test_data, derive_task=ld.create_synthetic(7000))
    if config_to_use.sequencing == 'annoy':
        test_data = ld.DerivedRangeDetectorFiles(create_from=test_data,
                                                 derive_task=ld.create_annoy_sequences(config_to_use.bucket_parameter))
    test_data = ld.DerivedRangeDetectorFiles(create_from=test_data, derive_task=ld.CreateAngles)
    if config_to_use.sequencing == 'theta':
        test_data = ld.DerivedRangeDetectorFiles(create_from=test_data,
                                                 derive_task=ld.create_sequences(config_to_use.bucket_parameter))
    if config_to_use.sequencing == 't/p':
        test_data = ld.DerivedRangeDetectorFiles(create_from=test_data, derive_task=ld.create_sequences(14))
        test_data = ld.DerivedRangeDetectorFiles(create_from=test_data,
                                                  derive_task=ld.create_phi_sequences(4))
    test_data = ld.DerivedRangeDetectorFiles(create_from=test_data, derive_task=ld.CreateNormalized)

    do_load = False
    estimator = PredictionMultiVolumeTracker(config_to_use, do_load=do_load)

    estimator.train(train_data, test_data)
