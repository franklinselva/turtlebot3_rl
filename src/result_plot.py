#!/usr/bin/env python

import rospy
import pyqtgraph as pg
import sys
import pickle
from std_msgs.msg import Float32
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Result")
        self.setGeometry(50, 50, 600, 350)
        self.graph_sub = rospy.Subscriber('result', Float32, self.data)
        self.ep = []
        self.rewards = []
        self.x = []
        self.count = 1
        self.size_ep = 0
        load_data = False

        if load_data:
            self.ep, self.data = self.load_data()
            self.size_ep = len(self.ep)
        self.plot()

    def data(self, data):
        self.ep.append(self.size_ep + self.count)
        self.count += 1
        self.rewards.append(data.data)

    def plot(self):
        self.rewardsPlt = pg.PlotWidget(self, title="Total reward")
        self.rewardsPlt.move(0, 10)
        self.rewardsPlt.resize(600, 300)

        self.timer2 = pg.QtCore.QTimer()
        self.timer2.timeout.connect(self.update)
        self.timer2.start(100)

        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.rewardsPlt.plot(self.ep, self.rewards, pen=(255, 0, 0))
        # self.save_data([self.ep, self.data])

    def load_data(self):
        try:
            with open("graph.txt") as f:
                x, y = pickle.load(f)
        except:
            x, y = [], []
        return x, y

    def save_data(self, data):
        with open("graph.txt", "wb") as f:
            pickle.dump(data, f)


def run():
        rospy.init_node('plot_graph')
        app = QApplication(sys.argv)
        GUI = Window()
        sys.exit(app.exec_())

run()