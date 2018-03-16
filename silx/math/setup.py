# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "16/03/2018"

import os.path

import numpy

from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('math', parent_package, top_path)
    config.add_subpackage('test')
    config.add_subpackage('fit')
    config.add_subpackage('medianfilter')

    # =====================================
    # histogramnd
    # =====================================
    histo_src = [os.path.join('histogramnd', 'src', 'histogramnd_c.c'),
                 'chistogramnd.pyx']
    histo_inc = [os.path.join('histogramnd', 'include'),
                 numpy.get_include()]

    config.add_extension('chistogramnd',
                         sources=histo_src,
                         include_dirs=histo_inc,
                         language='c')

    # =====================================
    # histogramnd_lut
    # =====================================
    config.add_extension('chistogramnd_lut',
                         sources=['chistogramnd_lut.pyx'],
                         include_dirs=histo_inc,
                         language='c')
    # =====================================
    # marching cubes
    # =====================================
    mc_src = [os.path.join('marchingcubes', 'mc_lut.cpp'),
              'marchingcubes.pyx']
    config.add_extension('marchingcubes',
                         sources=mc_src,
                         include_dirs=['marchingcubes', numpy.get_include()],
                         language='c++')

    # =====================================
    # marching squares
    # =====================================
    config.add_extension('marchingsquares',
                         sources=['marchingsquares.pyx'],
                         include_dirs=[numpy.get_include()],
                         language='c++',
                         extra_compile_args=["-std=c++11"],
                         extra_link_args=["-std=c++11"])
    config.add_extension('marchingsquares_mp',
                         sources=['marchingsquares_mp.pyx'],
                         include_dirs=[numpy.get_include()],
                         language='c++',
                         extra_compile_args=["-std=c++11", "-fopenmp"],
                         extra_link_args=["-std=c++11", "-fopenmp"])

    # min/max
    config.add_extension('combo',
                         sources=['combo.pyx'],
                         include_dirs=['include'],
                         language='c')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
