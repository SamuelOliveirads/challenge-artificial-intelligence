import pickle


def save_to_file(obj: any, filename: str) -> None:
    """Save an object to a file using pickle.

    Parameters
    ----------
    obj : any
        The object to be saved.
    filename : str
        The filename where the object will be saved.
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_from_file(filename: str) -> any:
    """Load an object from a file using pickle.

    Parameters
    ----------
    filename : str
        The filename where the object is saved.

    Returns
    -------
    any
        The object loaded from the file.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)
