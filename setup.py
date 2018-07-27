from setuptools import setup, find_packages

import imp

version = imp.load_source('autopool.version', 'autopool/version.py')
description='Adaptive pooling operators for multiple instance learning'

with open('README.md') as file:
    long_description = file.read()

setup(
    name='autopool',
    version=version.version,
    description=description,
    author='MARL',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/marl/autopool',
    download_url='http://github.com/marl/autopool/releases',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='deep learning, multiple instance learning, machine learning',
    license='MIT',
    install_requires=[
        'keras>=2.0',
    ],
    extras_require={ 'docs': ['numpydoc', 'sphinx-gallery'] }
)
