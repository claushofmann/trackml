import luigi
from trackml import dataset
from data_transformations.compute_angles import make_angles_file
from data_transformations.normalize import make_normalized_file
from data_transformations.compute_sequences import make_sequences_file
from data_transformations.compute_sequences_annoy import make_sequences_annoy_file
from data_transformations.synthetic import make_synthetic_file
import os
import multiprocessing

detector_data_root = 'specify/path/to/data'

# Overwrite default value with environment variable if present
try:
    detector_data_root = os.environ['trackml_data']
    print('Using {} for data files'.format(detector_data_root))
except:
    print('No environment variable set for data, using default {}'.format(detector_data_root))


def set_data_directory(directory):
    global detector_data_root
    detector_data_root = directory


def get_event_name(event_id):
    return 'event' + str(event_id).zfill(9)


def number_to_prefix(number, dir):
    """
    Create the file prefix from the number specified

    :param number: Event ID
    :return: Prefix of the corresponding file
    """
    number_str = str(number)
    file_number = number_str.zfill(9)
    file_prefix = dir + '/event' + file_number
    return file_prefix


class _DetectorFiles(luigi.Task):

    built = False

    def build(self):
        if not self.built:
            succeeded = luigi.build([self], local_scheduler=True)
            if not succeeded:
                raise Exception('Could not build needed files. Did you specify the path to the files correctly?')
            self.built = True


    def get_base_path(self):
        pass

    def get_event_id(self):
        pass

    def get_file_prefix(self):
        return number_to_prefix(self.get_event_id(), self.get_base_path())

    def do_load(self):
        return dataset.load_event(self.get_file_prefix())

    def load(self):
        self.build()
        return self.do_load()

    def do_load_hits(self):
        return dataset.load_event_hits(self.get_file_prefix())

    def do_load_truth(self):
        return dataset.load_event_truth(self.get_file_prefix())

    def do_load_particles(self):
        return dataset.load_event_particles(self.get_file_prefix())

    def do_load_cells(self):
        return dataset.load_event_cells(self.get_file_prefix())

    def output(self):
        base = self.get_file_prefix()
        return (luigi.LocalTarget(base + '-cells.csv'),
                luigi.LocalTarget(base + '-truth.csv'),
                luigi.LocalTarget(base + '-hits.csv'),
                luigi.LocalTarget(base + '-particles.csv'))


class RootDetectorFiles(_DetectorFiles):
    event_id = luigi.Parameter()

    def get_base_path(self):
        return detector_data_root

    def get_event_id(self):
        return self.event_id


class DerivedDetectorFiles(_DetectorFiles):
    create_from = luigi.Parameter()

    def get_path_extension(self):
        pass

    def get_event_id(self):
        return self.create_from.get_event_id()

    def get_base_path(self):
        return self.create_from.get_base_path() + self.get_path_extension()

    def requires(self):
        return self.create_from

    def create_dir(self):
        if not os.path.exists(self.get_base_path()):
            os.makedirs(self.get_base_path())


class CreateAngles(DerivedDetectorFiles):
    def get_path_extension(self):
        return '/angles'

    def run(self):
        self.create_dir()
        hits, cells, particles, truth = self.create_from.do_load()
        make_angles_file(self.get_file_prefix(), hits, cells, particles, truth)


class CreateNormalized(DerivedDetectorFiles):
    def get_path_extension(self):
        return '/normalized'

    def run(self):
        self.create_dir()
        hits, cells, particles, truth = self.create_from.do_load()
        make_normalized_file(self.get_file_prefix(), hits, cells, particles, truth)

def create_sequences(no_per_bucket):
    class CreateSequences(DerivedDetectorFiles):
        def get_path_extension(self):
            return '/sequences-{}'.format(no_per_bucket)

        def run(self):
            self.create_dir()
            hits, cells, particles, truth = self.create_from.do_load()
            make_sequences_file(self.get_file_prefix(), hits, cells, particles, truth, no_per_bucket)
    return CreateSequences

def create_phi_sequences(no_per_bucket):
    class CreatePhiSequences(DerivedDetectorFiles):
        def get_path_extension(self):
            return '/phi-sequences-{}'.format(no_per_bucket)

        def run(self):
            self.create_dir()
            hits, cells, particles, truth = self.create_from.do_load()
            make_sequences_file(self.get_file_prefix(), hits, cells, particles, truth, no_per_bucket, feature='phi')
    return CreatePhiSequences

def create_annoy_sequences(no_sequences):
    class CreateAnnoySequences(DerivedDetectorFiles):
        def get_path_extension(self):
            return '/annoy_sequences-{}'.format(no_sequences)

        def run(self):
            self.create_dir()
            hits, cells, particles, truth = self.create_from.do_load()
            make_sequences_annoy_file(self.get_file_prefix(), hits, cells, particles, truth, no_sequences)
    return CreateAnnoySequences

def create_synthetic(no_particles):
    class CreateSynthetic(DerivedDetectorFiles):
        def get_path_extension(self):
            return '/synthetic-{}'.format(no_particles)

        def run(self):
            self.create_dir()
            make_synthetic_file(self.get_file_prefix(), no_particles)
    return CreateSynthetic


derive_task_dict = {
    'norm': CreateNormalized,
    'phi': CreateAngles
}

class DetectorFile(_DetectorFiles):
    # Enter Tasks like this: 'norm+phi'
    derive_tasks = luigi.Parameter()
    event_id = luigi.Parameter()

    def __init__(self, derive_tasks=None, event_id=None):
        super().__init__(derive_tasks=derive_tasks, event_id=event_id)

    def get_base_path(self):
        return self.root.get_base_path()

    def get_event_id(self):
        return self.root.get_event_id()

    def requires(self):
        pass


class RangeDetectorFiles(luigi.Task):
    built = False

    def build(self):
        if not self.built:
            succeeded = luigi.build([self], local_scheduler=True, workers=multiprocessing.cpu_count())
            if not succeeded:
                raise Exception('Could not build needed files. Did you specify the path to the files correctly?')
            self.built = True

    def get_start_range(self):
        pass

    def get_end_range(self):
        pass

    def do_load_idx(self, idx):
        if self.get_length() > idx:
            return self.requires()[idx].do_load()
        else:
            raise Exception('Index not within range')

    def get_length(self):
        return int(self.get_end_range()) - int(self.get_start_range())

    def load(self):
        self.build()
        return self.do_load()

    def do_load(self):
        for file in self.requires():
            yield file.do_load()

    def batch_load(self, batch_size):
        self.build()
        return self.batch_do_load(batch_size)

    def get_all_files(self):
        self.build()
        return self.requires()

    def batch_do_load(self, batch_size):
        for batch in self.do_get_all_files_batched(batch_size):
            batch_loaded = (file.do_load() for file in batch)
            return batch_loaded

    def do_get_all_files_batched(self, batch_size):
        def get_batch(start_batch, end_batch):
            return self.requires()[start_batch:end_batch]
        for i in range(0, self.get_end_range() - self.get_start_range(), batch_size):
            yield get_batch(i, i + batch_size)

    def get_all_files_batched(self, batch_size):
        self.build()
        return self.do_get_all_files_batched(batch_size)

    def output(self):
        return [requ.output() for requ in self.requires()]


class RootRangeDetectorFiles(RangeDetectorFiles):
    start_range = luigi.Parameter()
    end_range = luigi.Parameter()

    def requires(self):
        return [RootDetectorFiles(str(i)) for i in range(self.start_range, self.end_range)]

    def get_start_range(self):
        return self.start_range

    def get_end_range(self):
        return self.end_range


class DerivedRangeDetectorFiles(RangeDetectorFiles):
    derive_task = luigi.Parameter()
    create_from = luigi.Parameter()

    def requires(self):
        return [self.derive_task(create_from=task) for task in self.create_from.requires()]

    def get_start_range(self):
        return self.create_from.get_start_range()

    def get_end_range(self):
        return self.create_from.get_end_range()

