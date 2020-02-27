import json
import pickle


def save_json(obj, path):
    with open(path, 'w') as f:
        return json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
