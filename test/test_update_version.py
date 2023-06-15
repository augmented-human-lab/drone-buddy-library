import re

init_file_paths = [r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\configurations\__init__.py',
                   r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\offline\atoms\__init__.py',
                   r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\offline\__init__.py',
                   r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\online\atoms\__init__.py',
                   r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\__init__.py',
                   r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\utils\__init__.py']

setup_file_path = r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\setup.py'


def change_init_version(new_version):
    for file_path in init_file_paths:
        with open(file_path, 'r') as file:
            content = file.read()
        modified_content = re.sub(r"__version__ = \"\d+\.\d+\.\d+\"", '__version__ = \"' + new_version + '\"', content)
        with open(file_path, 'w') as file:
            file.write(modified_content)


def change_setup_version(new_version):
    with open(setup_file_path, 'r') as file:
        content = file.read()
    modified_content = re.sub(r"version=['\"]\d+\.\d+\.\d+['\"]", 'version=\'' + new_version + '\'', content)
    with open(setup_file_path, 'w') as file:
        file.write(modified_content)


if __name__ == '__main__':
    new_version = '1.0.5'
    change_init_version(new_version)
    change_setup_version(new_version)
