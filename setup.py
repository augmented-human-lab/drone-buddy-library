from setuptools import find_packages, setup

setup(
    name='dronebuddylib',
    packages=find_packages(include=['dronebuddylib']),
    version='0.1.1',
    description='Everything to control and customize Tello',
    author='Me',
    license='MIT',
    install_requires=['numpy', 'progressbar', 'requests', 'pillow', 'imageio', 'imutils'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
