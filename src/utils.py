import pickle


def save_to_file(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_from_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
