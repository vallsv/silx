
import numpy

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.tools.CurveLegendsWidget import CurveLegendsWidget
from silx.gui.widgets.BoxLayoutDockWidget import BoxLayoutDockWidget
from silx.gui.plot.backends import BackendBase

from PyQt5 import Qt
from PyQt5.QtCore import QPointF
from PyQt5 import QtChart


class Marker(Qt.QGraphicsSimpleTextItem):
    def __init__(self, chart: QtChart.QChart):
        Qt.QGraphicsItem.__init__(self, parent=chart)
        self.__chart = chart

    def updatePosition(self):
        pos = QPointF(self.__x, self.__y)
        pos = self.__chart.mapToPosition(pos)
        self.setPos(pos)

    def setDataPos(self, x, y):
        self.__x = x
        self.__y = y
        self.updatePosition()


class BackendQtGraph(BackendBase.BackendBase, QtChart.QChartView):

    def __init__(self, plot, parent=None):
        QtChart.QChartView.__init__(self, parent=parent)
        BackendBase.BackendBase.__init__(self, plot, parent)
        self.__items = set([])

        chart = QtChart.QChart()
        chart.legend().setVisible(False)

        chart.createDefaultAxes()
        chart.setTitle("Prototype")
        self.setChart(chart)
        self.__view = self
        self.__chart = chart
        self.setRenderHint(Qt.QPainter.Antialiasing)
        self.setRubberBand(QtChart.QChartView.RectangleRubberBand)

    def getWidgetHandle(self):
        """Return the widget this backend is drawing to."""
        return self.__view

    def getQColor(self, color):
        from silx.gui import colors
        c = colors.rgba(color)
        return Qt.QColor.fromRgbF(c[0], c[1], c[2], c[3])

    def addCurve(self, x, y,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror, z, selectable,
                 fill, alpha, symbolsize):
        item = QtChart.QLineSeries(self)
        color = self.getQColor(color)
        linestyle = {
            '--': Qt.Qt.DashLine,
            '-': Qt.Qt.SolidLine,
            ':': Qt.Qt.DotLine,
            ';': Qt.Qt.DashDotLine,
        }[linestyle]

        pen = Qt.QPen(color, linewidth*1.5, linestyle)
        item.setPen(pen)

        for xpixel, ypixel in zip(x, y):
            item.append(xpixel, ypixel)
        self.__items.add(item)
        self.__chart.addSeries(item)
        self.__chart.createDefaultAxes()
        return item

    def addMarker__not_working(self, x, y, text, color,
                  selectable, draggable,
                  symbol, linestyle, linewidth, constraint, yaxis):
        item = QtChart.QLineSeries(self)
        item.setPointLabelsClipping(True)
        color = self.getQColor(color)
        item.setPointLabelsColor(color)
        item.setPointLabelsFormat(str(text))
        item.setPointLabelsVisible(True)
        item.setPointsVisible(True)
        item.append(x, y)
        self.__items.add(item)
        return item

    def addMarker(self, x, y, text, color,
                  selectable, draggable,
                  symbol, linestyle, linewidth, constraint, yaxis):
        item = Marker(self.__chart)
        item.setDataPos(x, y)
        item.setText(str(text))
        color = self.getQColor(color)
        item.setPen(Qt.QPen(color))
        self.__items.add(item)
        return item

    def remove(self, item):
        found = self.__items.pop(item, None)
        if found is not None:
            if isinstance(QChart.QAbstractSeries):
                self.__chart.removeSeries(item)
            elif isinstance(Marker):
                pass

    def __updateMarkerPosition(self):
        scene = self.scene()
        if scene is not None:
            for item in self.__items:
                if isinstance(item, Marker):
                    item.updatePosition()

    def resizeEvent(self, event: Qt.QResizeEvent):
        # In case of widget resize
        QtChart.QChartView.resizeEvent(self, event)
        # It have to be called after the super resize
        self.__updateMarkerPosition()

    def viewportEvent(self, event):
        # In case of zoom/pan
        result = QtChart.QChartView.viewportEvent(self, event)
        if result:
            # It have to be called after the super resize
            self.__updateMarkerPosition()
        return result

# First create the QApplication
app = qt.QApplication([])

# Create a plot and add some curves
backend=None
backend=BackendQtGraph
window = PlotWidget(backend=backend)
window.setWindowTitle("CurveLegendWidgets demo")

x = numpy.linspace(-numpy.pi, numpy.pi, 100)
window.addCurve(x, 2. * numpy.random.random(100) - 1.,
                legend='random',
                symbol='s', linestyle='--',
                color='red')
window.addCurve(x, numpy.sin(x),
                legend='sin',
                symbol='o', linestyle=':',
                color='blue')
window.addCurve(x, numpy.cos(x),
                legend='cos',
                symbol='', linestyle='-',
                color='blue')

window.addMarker(x=0, y=0, text="Foooo")


# Show the plot and run the QApplication
window.setAttribute(qt.Qt.WA_DeleteOnClose)
window.show()
window.resize(400, 300)

app.exec_()
