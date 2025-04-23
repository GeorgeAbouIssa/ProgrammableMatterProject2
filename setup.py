from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ConnectedMatterAgent",
        ["ConnectedMatterAgent.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "Visualizer",
        ["Visualizer.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "Controller",
        ["Controller.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="ConnectedMatterAgent",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)