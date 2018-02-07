

import os
import numpy


class DataSet:
    """
    A representation of a DataSet consisting of a train set and a test set
    """

    def __init__(self, dataset_dir, dataset_name):
        """
        Creates the dataset. The files consisting of the dataset must have the following format:
        Train_<dataset_name>.csv
        Test_<dataset_name>.csv
        :param dataset_dir: The directory where the datasets are located
        :param dataset_name: The dataset's name
        """
        train_db_path = os.path.join(dataset_dir, 'Train_' + dataset_name + '.csv')
        test_db_path = os.path.join(dataset_dir, 'Test_' + dataset_name + '.csv')
        db_paths = [train_db_path, test_db_path]
        contents = {}

        for db_path in db_paths:
            cur_content = []
            with open(db_path) as csv_file:
                read_lines = csv_file.readlines()

            for row in read_lines:
                row = row.strip().split(sep=',')
                row = list(map(float, row))
                cur_content.append(row)
            contents[db_path] = numpy.array(cur_content, dtype=float)

        self._train = contents[train_db_path]
        self._test = contents[test_db_path]
        self._batch_idx = 0 # Index pointing the next sample of the train set which should be taken with the next call of self.next_batch

    @property
    def train(self):
        """
        Returns a numpy 2D matrix which represents the train set as it is displayed within the .csv file
        :return: The train set
        """
        return self._train

    @property
    def test(self):
        """
        Returns a numpy 2D matrix which represents the test set as it is displayed within the .csv file
        :return: The test set
        """
        return self._test

    def _next_slice(self, slice_idx, slice_size):
        """
        Returns a numpy 2D matrix which has up to slice_size rows. These rows are taken from the self.train matrix and
        are consecutive starting from the slice_idx. If the end of the train set is to be reached, then
        min(slice_size, remaining) samples are returned.
        :param slice_idx: The starting index (inclusive)
        :param slice_size: The slice_size. Must not be greater than the length of the train set
        :return: A 2D numpy matrix from the train set with up to slice_size samples
        """
        assert (0 <= slice_size <= len(self._train))
        assert (0 <= slice_idx < len(self._train))

        slice_size = min(slice_size, (len(self._train) - slice_idx))
        return self._train[slice_idx:slice_idx+slice_size, :]

    def next_batch(self, batch_size):
        """
        Returns a numpy 2D matrix which has batch_size rows. These rows are taken from the self.train matrix and are
        consecutive. Each call returns the next batch_size rows. If the end of the train set is to be reached, a shuffle
        of the train set occurs and then the next batch is fetched.
        :param batch_size: The batch_size. Must not be greater than the length of the train set
        :return: A 2D numpy matrix from the train set with batch_size samples
        """
        assert(batch_size <= len(self._train))

        train_slice = self._next_slice(self._batch_idx, batch_size)
        slice_samples = train_slice.shape[0]
        if slice_samples < batch_size:
            numpy.random.shuffle(self._train)
            train_slice = numpy.concatenate((train_slice, self._next_slice(0, batch_size-slice_samples)), axis=0)

        self._batch_idx = (self._batch_idx + batch_size) % len(self._train)
        return train_slice

    def normalize(self):
        """
        Normalizes data of both the train and the test set in range [0, 1]
        :return: -
        """
        # The normalization for each sample is done as:
        # z = (x - min(x)) / (max(x) - min(x)) where x is the sample and z is the normalized sample
        datasets = {'train':self._train, 'test':self._test}
        samples_min = {}
        samples_denominator = {}
        for key, dataset in datasets.items():
            min = numpy.amin(dataset, axis=1) # minimum value for each sample in the current dataset. [samples, 1]
            max = numpy.amax(dataset, axis=1) # maximum value for each sample in the current dataset
            samples_min[key] = min
            samples_denominator[key] = numpy.subtract(max, min)

        for key, dataset in datasets.items():
            for i in range(0, dataset.shape[0]):
                dataset[i, :] = numpy.divide(numpy.subtract(dataset[i, :], samples_min[key][i]), samples_denominator[key][i])
        return
