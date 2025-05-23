from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compile Cython extension with optimized flags and disable deprecated NumPy API
ext_modules = cythonize([
    Extension(
        name="cython_attention",
        sources=["cython_attention.pyx"],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[
            "-O3",             # High-level optimizations
            "-march=native",   # Use native CPU instructions
            "-std=c++11"       # C++11 standard
        ],
        include_dirs=[np.get_include()],
    )
],
compiler_directives={
    "boundscheck": False,    # Disable index bounds checking
    "wraparound": False,     # Disable negative index wraparound
    "cdivision": True,       # Enable C division semantics
    "language_level": "3"   # Target Python 3 code
},
annotate=True)            # Generate annotation HTML

setup(
    name="cython_attention",
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)
