# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""Basic tests for Plot"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/04/2017"


import unittest
import numpy
from silx.gui import qt
from .. import items
from .. import PlotEvents


class TestEvents(unittest.TestCase):
    """Test object events"""

    def testLimitsChangedEvent(self):
        sourceObj = "a"
        xRange = "b"
        yRange = "c"
        y2Range = "d"
        event = PlotEvents.LimitsChangedEvent(
            sourceObj, xRange, yRange, y2Range)
        self.assertEquals(event.getType(), PlotEvents.Type.LimitChanged)
        self.assertEquals(event.getXRange(), xRange)
        self.assertEquals(event.getYRange(), yRange)
        self.assertEquals(event.getY2Range(), y2Range)

    def testMouseEvent(self):
        event = "e"
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseEvent(event, button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), "e")
        self.assertEquals(event.getButton(), "b")
        self.assertEquals(event.getScenePos()[0], xData)
        self.assertEquals(event.getScenePos()[1], yData)
        self.assertEquals(event.getScreenPos()[0], xPixel)
        self.assertEquals(event.getScreenPos()[1], yPixel)

    def testMouseClickedEvent(self):
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseClickedEvent(button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.MouseClicked)

    def testMouseDoubleClickedEvent(self):
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseDoubleClickedEvent(button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.MouseDoubleClicked)

    def testMouseMovedEvent(self):
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseMovedEvent([xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.MouseMoved)
        self.assertEquals(event.getButton(), qt.Qt.NoButton)

    def testItemClickedEvent(self):
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        item = "foo"
        itemIndices = [10]
        event = PlotEvents.ItemClickedEvent(button, item, itemIndices, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.ItemClicked)
        self.assertEquals(event.getButton(), "b")
        self.assertEquals(event.getScenePos()[0], xData)
        self.assertEquals(event.getScenePos()[1], yData)
        self.assertEquals(event.getScreenPos()[0], xPixel)
        self.assertEquals(event.getScreenPos()[1], yPixel)
        self.assertEquals(event.getItem(), item)
        self.assertEquals(event.getItemIndices(), itemIndices)

    def testItemRegionChangeEvent(self):
        item = "foo"
        eventType = "e"
        event = PlotEvents.ItemRegionChangeEvent(eventType, item)
        self.assertEquals(event.getType(), eventType)
        self.assertEquals(event.getItem(), item)

    def testItemRegionChangeFinishedEvent(self):
        event = PlotEvents.ItemRegionChangeFinishedEvent("item")
        self.assertEquals(event.getType(), PlotEvents.Type.RegionChangeFinished)

    def testItemRegionChangeStrartedEvent(self):
        event = PlotEvents.ItemRegionChangeStartedEvent("item")
        self.assertEquals(event.getType(), PlotEvents.Type.RegionChangeStarted)

    def testItemRegionChangedEvent(self):
        event = PlotEvents.ItemRegionChangedEvent("item")
        self.assertEquals(event.getType(), PlotEvents.Type.RegionChanged)

    def testItemHoverdEvent(self):
        posData = [10, 11]
        posPixel = [12, 13]
        item = "a"
        event = PlotEvents.ItemHoveredEvent(item, posData, posPixel)
        self.assertEquals(event.getType(), PlotEvents.Type.ItemHovered)
        self.assertEquals(event.getItem(), item)
        self.assertEquals(event.getScenePos(), posData)
        self.assertEquals(event.getScreenPos(), posPixel)


class TestDictionaryLikeGetter(unittest.TestCase):
    """Test old getter in plot events. Events have to support a set of
    dictionnary-like getter"""

    def testDrawingProgressEvent(self):
        points = numpy.array([[-1, 0], [1, 1], [2, 2]])
        item = items.RectangleDrawItem()
        item.setData(points[:, 0], points[:, 1])
        event = PlotEvents.ItemRegionChangedEvent(item)

        self.assertEquals(event['event'], "drawingProgress")
        self.assertEquals(event['type'], "rectangle")
        numpy.testing.assert_almost_equal(event['points'], points)
        numpy.testing.assert_almost_equal(event['xdata'], points[:, 0])
        numpy.testing.assert_almost_equal(event['ydata'], points[:, 1])
        self.assertEquals(event['x'], -1)
        self.assertEquals(event['y'], 0)
        self.assertEquals(event['width'], 3)
        self.assertEquals(event['height'], 2)
        self.assertEquals(event['parameters'], {})

    def testDrawingFinishedEvent(self):
        points = numpy.array([[0, 1], [1, 1]])
        item = items.VLineDrawItem()
        item.setData(points[:, 0], points[:, 1])
        event = PlotEvents.ItemRegionChangeFinishedEvent(item)

        self.assertEquals(event['event'], "drawingFinished")
        self.assertEquals(event['type'], "vline")
        numpy.testing.assert_almost_equal(event['points'], points)
        numpy.testing.assert_almost_equal(event['xdata'], points[:, 0])
        numpy.testing.assert_almost_equal(event['ydata'], points[:, 1])
        self.assertEquals(event['parameters'], {})

    def testMouseEvent(self):
        button = qt.Qt.LeftButton
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseClickedEvent(button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event['event'], "mouseClicked")
        self.assertEquals(event['button'], "left")
        self.assertEquals(event['x'], xData)
        self.assertEquals(event['y'], yData)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testHoverEvent(self):
        label = "a"
        posData = [10, 11]
        posPixel = [12, 13]
        item = items.YMarker()
        item._setLegend(label)
        item._setSelectable(True)
        item._setDraggable(False)
        event = PlotEvents.ItemHoveredEvent(item, posData, posPixel)
        self.assertEquals(event['event'], "hover")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['x'], posData[0])
        self.assertEquals(event['y'], posData[1])
        self.assertEquals(event['xpixel'], posPixel[0])
        self.assertEquals(event['ypixel'], posPixel[1])
        self.assertEquals(event['draggable'], False)
        self.assertEquals(event['selectable'], True)

    def testImageEvent(self):
        button = qt.Qt.LeftButton
        label = "b"
        eventType = "image"
        col = "d"
        row = "e"
        x = "f"
        y = "g"
        xPixel = "h"
        yPixel = "i"
        image = items.ImageBase()
        image._setLegend(label)
        event = PlotEvents.ItemClickedEvent(
            button,
            image,
            [(row, col)],
            (x, y),
            (xPixel, yPixel))
        self.assertEquals(event['event'], "imageClicked")
        self.assertEquals(event['button'], "left")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], eventType)
        self.assertEquals(event['col'], col)
        self.assertEquals(event['row'], row)
        self.assertEquals(event['x'], x)
        self.assertEquals(event['y'], y)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testCurveEvent(self):
        button = qt.Qt.RightButton
        label = "b"
        eventType = "curve"
        xData2 = 11
        yData2 = 12
        x = "f"
        y = "g"
        xPixel = "h"
        yPixel = "i"
        index = 2
        xData = [0, 0, xData2, 0, 0]
        yData = [0, 0, yData2, 0, 0]
        curve = items.Curve()
        curve._setLegend(label)
        curve.setData(xData, yData)
        event = PlotEvents.ItemClickedEvent(
            button,
            curve,
            index,
            (x, y),
            (xPixel, yPixel))
        self.assertEquals(event['event'], "curveClicked")
        self.assertEquals(event['button'], "right")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], eventType)
        self.assertEquals(event['xdata'], xData2)
        self.assertEquals(event['ydata'], yData2)
        self.assertEquals(event['x'], x)
        self.assertEquals(event['y'], y)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testLimitsChangedEvent(self):
        sourceObj = "a"
        xRange = "b"
        yRange = "c"
        y2Range = "d"
        event = PlotEvents.LimitsChangedEvent(
            sourceObj, xRange, yRange, y2Range)
        self.assertEquals(event['event'], "limitsChanged")
        self.assertEquals(event['source'], id(sourceObj))
        self.assertEquals(event['xdata'], xRange)
        self.assertEquals(event['ydata'], yRange)
        self.assertEquals(event['y2data'], y2Range)

    def testMarkerMovingEvent(self):
        eventType = "markerMoving"
        label = "l"
        draggable = True
        selectable = False
        posDataMarker = [10, 11]
        posPixelCursor = [12, 13]
        posDataCursor = [14, 15]

        item = items.marker.Marker()
        item._setLegend(label)
        item.setPosition(posDataMarker[0], posDataMarker[1])
        item._setDraggable(draggable)
        item._setSelectable(selectable)

        event = PlotEvents.ItemRegionChangedEvent(item, posDataCursor, posPixelCursor)
        self.assertEquals(event['event'], eventType)
        self.assertEquals(event['button'], "left")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], "marker")
        self.assertEquals(event['x'], posDataCursor[0])
        self.assertEquals(event['y'], posDataCursor[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])

    def testMarkerClicked(self):
        button = "left"
        label = "l"
        draggable = True
        selectable = False
        posDataMarker = (10, 11)
        posPixelCursor = (12, 13)
        scenePos = (14, 15)
        index = "foo"
        item = items.marker.Marker()
        item._setLegend(label)
        item.setPosition(posDataMarker[0], posDataMarker[1])
        item._setDraggable(draggable)
        item._setSelectable(selectable)

        event = PlotEvents.ItemClickedEvent(button, item, index, scenePos, posPixelCursor)
        self.assertEquals(event['event'], "markerClicked")
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], "marker")
        self.assertEquals(event['x'], posDataMarker[0])
        self.assertEquals(event['y'], posDataMarker[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])

    def testMarkerMovingOld(self):
        eventType = "markerMoving"
        button = "b"
        label = "l"
        type_ = "t"
        draggable = "d"
        selectable = "s"
        posDataMarker = ["dm0", "dm1"]
        posPixelCursor = ["pc0", "pc1"]
        posDataCursor = ["dc0", "dc1"]
        event = PlotEvents.prepareMarkerSignal(
            eventType, button, label, type_, draggable, selectable,
            posDataMarker, posPixelCursor, posDataCursor)
        self.assertEquals(event['event'], eventType)
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], type_)
        self.assertEquals(event['x'], posDataCursor[0])
        self.assertEquals(event['y'], posDataCursor[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])

    def testMarkerClickedOld(self):
        eventType = "markerClicked"
        button = "b"
        label = "l"
        type_ = "t"
        draggable = "d"
        selectable = "s"
        posDataMarker = (10, 11)
        posPixelCursor = (12, 13)
        event = PlotEvents.prepareMarkerSignal(
            eventType, button, label, type_, draggable, selectable,
            posDataMarker, posPixelCursor, posDataCursor=None)
        self.assertEquals(event['event'], eventType)
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], type_)
        self.assertEquals(event['x'], posDataMarker[0])
        self.assertEquals(event['y'], posDataMarker[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestEvents))
    test_suite.addTest(loader(TestDictionaryLikeGetter))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
