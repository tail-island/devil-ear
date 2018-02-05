import numpy   as np
import os.path as path
import pickle

from funcy       import cat, count, juxt, map, mapcat, repeat
from keras.utils import to_categorical
from utility     import child_paths


def load_data(data_path='./data'):
    def is_x_sample_path(data_path):
        return data_path.endswith('.pickle')

    def load_x_sample(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def load_characters(data_path):
        return tuple(map(load_x_sample, filter(is_x_sample_path, child_paths(data_path))))

    def load_actors(data_path):
        return tuple(map(load_characters, filter(path.isdir, child_paths(data_path))))

    def to_x(x_sample_node):
        return np.array(tuple(cat(cat(x_sample_node))))

    def to_y(x_sample_nodes):
        def child_size(x_sample_node):
            if not isinstance(x_sample_node, tuple):
                return 1
            else:
                return sum(map(child_size, x_sample_node))

        return np.array(tuple(zip(*map(lambda x_sample_nodes: mapcat(lambda x_sample_node, i: repeat(i, child_size(x_sample_node)), x_sample_nodes, count()), (x_sample_nodes, cat(x_sample_nodes))))))

    def load_x_and_y(type):
        return juxt(to_x, to_y)(tuple(map(load_actors, filter(path.isdir, child_paths(path.join(data_path, type))))))

    return map(load_x_and_y, ('train', 'validate'))
