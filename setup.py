from setuptools import setup, find_packages

with open("dronebuddylib.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setup(
    name='dronebuddylib',
    version='1.0.7',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'dronebuddylib': ['resources/*'],
    },
    zip_safe=False,
    setup_requires=[
        'setuptools>=50.3.0',
    ],
    install_requires=[requirements],
    python_requires='>=3.9',
    description='Everything to control and customize Tello',
    author='Malsha de Zoysa',
    author_email='malsha@ahlab.org',
    license='MIT',
)
