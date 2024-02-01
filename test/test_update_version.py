import re

init_file_paths = [
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\bodyfeatureextraction\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\intentrecognition\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\facerecognition\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectdetection\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\speechgeneration\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\speechrecognition\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\textrecognition\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\configurations\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\models\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\utils\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\molecules\__init__.py',
    r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\exceptions\__init__.py',

    ]

setup_file_path = r'C:\Users\Public\projects\drone-buddy-library\setup.py'


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
    new_version = '2.0.25'
    change_init_version(new_version)
    change_setup_version(new_version)
