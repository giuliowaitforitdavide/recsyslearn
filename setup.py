#!/usr/bin/env python

"The setup script."

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "pandas"]

test_requirements = ["numpy", "pandas"]

setup(
    author="Giulio Davide Carparelli",
    author_email="giulio.davide.97@gmail.com",
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.10",
    ],
    description="A library to compute fairness of recommender systems.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords="recsyslearn",
    name="recsyslearn",
    packages=find_packages(include=["recsyslearn", "recsyslearn.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/giuliowaitforitdavide/recsyslearn",
    version="2.0.2",
    zip_safe=False,
)
