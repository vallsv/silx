#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""
Example displaying a cyclic pattern.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui import colors


def create_sine_ramp_ring(size):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    angle = numpy.angle(yy + xx * 1j) + 3 * numpy.pi
    distance = numpy.sqrt(yy**2 + xx**2)

    r_max = half
    r_min = half / 2

    # Coefs came from Nicolás Guarín-Zapata
    # URL: https://nicoguaro.github.io/posts/cyclic_colormaps/

    r_norm = (distance - r_min) / (r_max - r_min)
    data = (0.1 * 2 * numpy.pi) * r_norm**2 * numpy.sin(angle * 50) + angle
    data = numpy.mod(data, 2 * numpy.pi)
    mask = numpy.logical_or(distance > r_max, distance < r_min)
    data[mask] = float("nan")

    return data


def create_ring(size):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    angle = numpy.angle(yy + xx * 1j)
    distance = numpy.sqrt(yy**2 + xx**2)

    ring = angle.copy()
    mask = numpy.logical_or(distance > half, distance < half / 2)
    ring[mask] = float("nan")

    return data


if __name__ == "__main__":
    app = qt.QApplication([])

    # data = create_ring(1024)
    data = create_sine_ramp_ring(1024)
    colormap = colors.Colormap(name="hsv")
    plot = Plot2D()
    plot.setDefaultColormap(colormap)
    plot.addImage(data=data, colormap=colormap)
    plot.show()

    app.exec_()
