from serde.yaml import to_yaml, from_yaml

def save_to_yaml(obj, path: str):
    """Description. Save serializable object to yaml."""

    with open(path, "w") as file:
        file.write(to_yaml(obj))

def read_yaml(obj_type, path: str): 
    """Description. Read yaml file and load it into python object."""

    with open(path, "r") as file:
        data = file.read()
    
    return from_yaml(c=obj_type, s=data)