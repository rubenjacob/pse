from setuptools import setup, find_packages

setup(
    name="pse",
    author="Ruben Jacob",
    author_email="rubenjacob@outlook.com",
    packages=find_packages(exclude=['test_pse'])
    # install dependencies separately using conda
)
