#!/usr/bin/env python

'The setup script.'

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'pandas']

test_requirements = ['numpy', 'pandas', 'unittest']

setup(
    author='Giulio Davide Carparelli',
    author_email='giulio.davide.97@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description='A small library to compute fairness of recommender systems.',
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='recsyslearn',
    name='recsyslearn',
    packages=find_packages(include=['recsyslearn', 'recsyslearn.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/giuliowaitforitdavide/recsyslearn',
    version='0.1.0',
    zip_safe=False,
)
