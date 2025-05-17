from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize([
    Extension(
        name="cython_attention",
        sources=["cython_attention.pyx"],
        langage = "c++",
    )
])

setup(
    name="cython_attention",
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)
