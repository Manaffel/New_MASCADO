#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='New_MASCADO',
    version='1.0',
    description='Expansion for MICADO.',
    author='Jonas Sauter',
    author_email='jonas.sauter@web.de',
    python_requires='>=3.5',
    url='https://github.com/Manaffel/New_MASCADO',
    license='GPLv3',

    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=[
        'numpy>=1.14',
        'matplotlib',
        'scipy',
        'pandas'],
    scripts=['scripts/mascado_analyze',
        'scripts/mascado_compare'],
    test_suite='tests',
)
