from setuptools import setup, find_packages

setup(
    name='dronebuddylib',
    version='0.2.00',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'dronebuddylib': ['atoms/resources/*'],
    },
    zip_safe=False,
    setup_requires=[
        'setuptools>=50.3.0',
    ],
    install_requires=['numpy', 'requests', 'pillow', 'opencv-python', 'pyttsx3'],
    description='Everything to control and customize Tello',
    author='Malsha de Zoysa',
    author_email='malsha@ahlab.org',
    license='MIT',
)
