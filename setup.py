#!/usr/bin/python
# coding: utf8

__author__ = "Jérôme Kieffer"
__date__ = "27/11/2015"
__license__ = "MIT"

import sys
import os
try:
    from setuptools import setup
    from setuptools.command.build_py import build_py as _build_py
except ImportError:
    from distutils.core import setup
    from distutils.command.build_py import build_py as _build_py


PROJECT = "silx"
cmdclass = {}


def get_version():
    import version
    return version.strictversion


def get_readme():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, "README.md"), "r") as fp:
        long_description = fp.read()
    return long_description


classifiers = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Environment :: MacOS X",
               "Environment :: Win32 (MS Windows)",
               "Environment :: X11 Applications :: Qt",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Natural Language :: English",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX",
               "Programming Language :: Cython",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Documentation :: Sphinx",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Software Development :: Libraries :: Python Modules",
               ]


class build_py(_build_py):
    """
    Enhanced build_py which copies version to the built
    """
    def build_package_data(self):
        """Copy data files into build directory
        Patched in such a way version.py -> silx/_version.py"""
        print(self.data_files)
        _build_py.build_package_data(self)
        for package, src_dir, build_dir, filenames in self.data_files:
            if package == PROJECT:
                filename = "version.py"
                target = os.path.join(build_dir, "_" + filename)
                self.mkpath(os.path.dirname(target))
                self.copy_file(os.path.join(filename), target,
                               preserve_mode=False)
                break

cmdclass['build_py'] = build_py

install_requires = ["numpy", "h5py"]
setup_requires = ["numpy", "cython"]


setup(name=PROJECT,
      version=get_version(),
      url="https://github.com/silex-kit/silx",
      author="data analysis unit",
      author_email="silx@esrf.fr",
      classifiers=classifiers,
      description="Software library for X-Ray data analysis",
      long_description=get_readme(),
      packages=["silx", "silx.io", "silx.third_party", "silx.visu"],
      install_requires=install_requires,
      setup_requires=setup_requires,
      cmdclass=cmdclass,
      )
