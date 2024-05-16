from dronebuddylib.utils.utils import logger


def read_from_file_and_return_index(file_path):
    """Read an integer from a file."""
    try:
        with open(file_path, 'r') as file:  # 'r' mode is for reading
            return int(file.read().strip())
    except IOError:
        logger.log_error("Utils", "Error while reading from the file : " + file_path)
        return -1


def update_file_with_index(file_path, index):
    """Write an integer to a file."""
    try:
        with open(file_path, 'w') as file:  # 'w' mode will create the file if it doesn't exist
            file.write(str(index))
            return True
    except IOError:
        logger.log_error("Utils", "Error while writing to the file : " + file_path)
        return False
