from setuptools import setup

setup(
    name='dronebuddylib',
    packages=['dronebuddylib'],
    version='0.1.7',
    description='Everything to control and customize Tello',
    author='Malsha de Zoysa',
    author_email='malsha@ahlab.org',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'progressbar', 'requests', 'pillow', 'imageio', 'imutils', 'opencv-python', 'pyttsx3'],
)
