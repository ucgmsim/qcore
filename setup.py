from os.path import join
from setuptools import setup

setup(
    name='qcore',
    version='1.1.4',
    packages=['qcore'],
    url='https://github.com/ucgmsim/qcore',
    description='QuakeCoRE Library',
    package_data={'qcore': [join('configs', '*.json')]},
    install_requires=['numpy', 'scipy>=0.16', 'dataclasses'],
    include_package_data=True,
)
