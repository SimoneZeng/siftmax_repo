from setuptools import Extension, setup
import numpy


extra_link_args = ["-fopenmp"]


ext_modules = [
    Extension(
        "rabitq_official_ext",
        sources=["rabitq_official_wrapper.cpp"],
        include_dirs=[
            numpy.get_include(),
            "RaBitQ-Library/include",
        ],
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-march=native",
            "-fopenmp",
        ],
        extra_link_args=extra_link_args,
    )
]


setup(
    name="rabitq_official_ext",
    version="0.1.0",
    ext_modules=ext_modules,
)
