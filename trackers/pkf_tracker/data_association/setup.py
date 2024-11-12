import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++14']

ext_modules = [
    Extension(
        'papy',
        ['ProbDataAssociation.cpp', 'wrap.cpp', 'permanent.cpp'],
        include_dirs=['.',
                      '/usr/include/eigen3', 
                      'pybind11/include'],
        language='c++',
        extra_compile_args = cpp_args,
    ),
]

setup(
    name='papy',
    version='0.0.1',
    author='Hanwen',
    ext_modules=ext_modules,
)

