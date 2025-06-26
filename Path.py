import pathlib

def get_full_path(path: str) -> pathlib.Path:
    """
    Returns a pathlib.Path object for the given path string.
    
    Args:
        path (str): The path string to convert.
        
    Returns:
        pathlib.Path: A Path object representing the given path.
    """
    di = pathlib.Path(__file__).parent.absolute()
    return pathlib.Path(di, path).resolve()