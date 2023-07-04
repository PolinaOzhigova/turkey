from setuptools import setup

import json
import os

def read_pipenv_dependencies(fname):
    filepath = os.path.join(os.path.dirname(__file__), fname)
    with open(filepath) as lockfile:
        lockjson = json.load(lockfile)
        return [dependency for dependency in lockjson.get('default')]

if __name__ == '__main__':
    setup(
        name='turkey',
        version='1.7',
        description='',
        license='MIT',
        url='https://github.com/PolinaOzhigova/turkey.git',
        packages=['turkey'],       
        install_requires=[
              *read_pipenv_dependencies('Pipfile.lock'),
        ],
        python_requires='>=3.10',
    )