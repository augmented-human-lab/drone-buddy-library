from setuptools import setup

setup(
    name='dronebuddylib',
    packages=['dronebuddylib'],
    version='0.1.6',
    description='Everything to control and customize Tello',
    author='Malsha de Zoysa',
    author_email='malsha@ahlab.org',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'progressbar', 'requests', 'pillow', 'imageio', 'imutils', 'opencv-python'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
