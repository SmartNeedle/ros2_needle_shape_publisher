import numpy as np
from matplotlib.lines import Line2D 
import matplotlib.pyplot as plt

from typing import (
    List,
    Dict,
)

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray

from .utilities import msg2poses

class NeedleShapePlotter(Node):
    def __init__(self, name="NeedleShapePlotter"):
        super().__init__(node_name=name)

        self.figure, self.axes = plt.subplots(ncols=1, nrows=2, sharex=True)
        self.axes: List[plt.Axes]

        self._plot_lines: Dict[str, Line2D] = dict()
        self.plotting                       = True

        self.sub_needleshape = self.create_subscription(
            PoseArray,
            "state/current_shape",
            self.sub_needleshape_callback,
            1,
        )

    # __init__
        
    @property
    def initialized(self):
        return len(self._plot_lines) > 0
        
    def plot_shape(self, position: np.ndarray, title: str=None):
        if not self.initialized:
            self.get_logger().info(f"Beginning to plot shape on topic: {self.sub_needleshape.topic_name}")
            plt.ion()
            
            self.plotting          = True
            self._plot_lines['xz'] = self.axes[0].plot(
                position[:, 2],
                position[:, 0],
                '*-',
            )[0]
            self._plot_lines['yz'] = self.axes[1].plot(
                position[:, 2],
                position[:, 1],
                '*-',
            )[0]

            self.figure.suptitle( f"Needle shape: {self.sub_needleshape.topic_name}" )
            self.axes[-1].set_xlabel( 'Z [mm] -> Insertion Direction' )
            
            self.axes[0].set_ylabel( 'Right <- X [mm] -> Left' )
            self.axes[1].set_ylabel( 'Bend Tip Down <- Y [mm] -> Bend Tip Up' )

            self.axes[0].set_title("Top View")
            self.axes[1].set_title("View from Right of Needle")

            plt.show()

            return
        
        # if

        if not plt.fignum_exists(self.figure.number):
            self.plotting = False
            return
        
        # if
        
        self._plot_lines['xz'].set_xdata(position[:, 2])
        self._plot_lines['xz'].set_ydata(position[:, 0])

        self._plot_lines['yz'].set_xdata(position[:, 2])
        self._plot_lines['yz'].set_ydata(position[:, 1])

        # update the axes limits
        xlim    = list(self.axes[0].get_xlim())
        xlim[0] = min(xlim[0], position[:, 2].min())
        xlim[1] = max(xlim[1], position[:, 2].max())
        ylims   = [list(ax.get_ylim()) for ax in self.axes]

        for i, ylim in enumerate(ylims):
            ylim[0] = min(ylim[0], position[:, i].min())
            ylim[1] = max(ylim[1], position[:, i].max())

            ylims[i] = ylim
        # for

        self.axes[0].set_xlim(xlim[0],     xlim[1])
        self.axes[0].set_ylim(ylims[0][0], ylims[0][1])
        self.axes[1].set_ylim(ylims[1][0], ylims[1][1])

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # plot_shape

        
        
    def sub_needleshape_callback(self, msg: PoseArray):
        poses = msg2poses(msg)
        pmat  = poses[:, :3, 3]

        self.get_logger().debug(f"Plotting Position at time: {msg.header.stamp}")
        self.plot_shape(pmat)

    # sub_needleshape_callback
        
# class: NeedleShapePlotter

def main(args=None):
    rclpy.init(args=args)

    plotter = NeedleShapePlotter()

    try:
        while plotter.plotting:
            rclpy.spin_once(plotter)

        # while
    except KeyboardInterrupt:
        pass

    finally:
        plotter.destroy_node()
        rclpy.shutdown()

# main
    
if __name__ == "__main__":
    main()

# if