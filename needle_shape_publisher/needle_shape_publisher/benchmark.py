import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile

# ROS2 packages
import rclpy
from rclpy.node import Node
from rclpy.parameter import ParameterValue
from rclpy.time import Time

# messages
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseArray, Point, PoseStamped
from std_msgs.msg import Float64MultiArray
from rcl_interfaces.srv import GetParameters

# current package
from . import shape_sensing_needle


class ShapeSensingBenchmarker(Node):
    NUM_SAMPLES = 100

    PARAM_NS          = "benchmarker"
    PARAM_ODIR        = f"{PARAM_NS}.odir"
    PARAM_NUMSAMPLES  = f"{PARAM_NS}.num_samples"
    PARAM_INSDEPTHINC = f"{PARAM_NS}.insertion_depth.increment"

    def __init__(self, name="ShapeSensingNeedleBenchmarkerNode"):
        super().__init__(name)

        # data storage
        self.num_aas       = None
        self.num_chs       = None
        self.needle_length = None

        self.current_insertion_depth     = 10.0
        self.current_insertion_depth_ts  = None
        self.current_insertion_depth_inc = 10.0

        self.benchmark_data = pd.DataFrame(
            columns=[
                "insertion_depth__mm",           # mm
                "insertion_depth_timestamp__ns", # nanoseconds
                "msg_timestamp__ns",             # nanoseconds
                "ros_timestamp__ns",             # nanoseconds
            ]
        )

        # parameters
        self.declare_parameter(
            ShapeSensingBenchmarker.PARAM_ODIR,
            tempfile.gettempdir(),
        )
        self.odir = self.get_parameter(ShapeSensingBenchmarker.PARAM_ODIR).get_parameter_value().string_value

        self.declare_parameter(
            ShapeSensingBenchmarker.PARAM_INSDEPTHINC,
            self.current_insertion_depth_inc,
        )
        self.current_insertion_depth_inc = self.get_parameter(ShapeSensingBenchmarker.PARAM_INSDEPTHINC).get_parameter_value().double_value
        
        self.declare_parameter(
            ShapeSensingBenchmarker.PARAM_NUMSAMPLES,
            ShapeSensingBenchmarker.NUM_SAMPLES,
        )
        self.num_samples = self.get_parameter(ShapeSensingBenchmarker.PARAM_NUMSAMPLES).get_parameter_value().integer_value

        # publishers
        self.pub_entrypoint = self.create_publisher(
            Point,
            'state/skin_entry',
            10,
        )
        self.pub_needlepose = self.create_publisher(
            PoseStamped,
            '/stage/state/needle_pose',
            10,
        )
        self.pub_curvatures = self.create_publisher(
            Float64MultiArray,
            'state/curvatures',
            10,
        ) # first attempt in curvature space
        self.pub_procsignals = self.create_publisher(
            Float64MultiArray,
            'sensors/processed',
            10,
        ) # if done in sensor space

        # subscribers
        self.sub_shape = self.create_subscription(
            PoseArray,
            'state/current_shape',
            self.sub_shape_callback,
            1
        )

        # timers
        self.timer_pub_curvatures = self.create_timer(0.001, self.publish_curvatures)
        self.timer_pub_stagepose  = self.create_timer(0.01,  self.publish_needlepose)
        self.timer_pub_entrypoint = self.create_timer(0.01,  self.publish_entrypoint)

    # __init__

    @property
    def completed(self):
        return self.current_insertion_depth > self.needle_length
    
    # property: completed

    @property
    def configured(self):
        return all(
            [
                self.num_aas is not None,
                self.num_chs is not None,
                self.needle_length is not None,
            ]
        )
    
    # property: configured

    def compute_time_deltas(self):
        """ Computes the time deltas for message received and processing """
        # compute the time deltas for statistical analysis
        self.benchmark_data = self.benchmark_data.assign(
            delta_msg_timestamp__ns=pd.NA,
            delta_ros_timestamp__ns=pd.NA,
        )

        # compute time differences from previous timestamp
        mask_neq_insertion_depth = np.append(
            [False],
            np.logical_not(
                self.benchmark_data["insertion_depth__mm"].to_numpy()[:-1]
                == self.benchmark_data["insertion_depth__mm"].to_numpy()[1:]
            )
        )

        self.benchmark_data.loc[1:, "delta_msg_timestamp__ns"] = (
            self.benchmark_data.loc[1:, "msg_timestamp__ns"]
            - self.benchmark_data["msg_timestamp__ns"].iloc[:-1].to_numpy()
        )
        self.benchmark_data.loc[1:, "delta_ros_timestamp__ns"] = (
            self.benchmark_data.loc[1:, "ros_timestamp__ns"]
            - self.benchmark_data["ros_timestamp__ns"].iloc[:-1].to_numpy()
        )

        self.benchmark_data.loc[
            mask_neq_insertion_depth,
            [
                "delta_msg_timestamp__ns",
                "delta_ros_timestamp__ns"
            ]
        ] = pd.NA # invalidate insertion depths that are the first

        # compute time differences for when the insertion depth received changed
        first_ts = self.benchmark_data.groupby("insertion_depth__mm").min()
        last_ts  = self.benchmark_data.groupby("insertion_depth__mm").max()

        diff_lenchange_ts = last_ts.iloc[:-1].to_numpy() - first_ts.iloc[1:]

        self.benchmark_data = self.benchmark_data.merge(
            diff_lenchange_ts["delta_msg_timestamp__ns"].rename("delta_lenchange_timestamp__ns"),
            on="insertion_depth__mm",
            how="left",
        )

    # compute_time_deltas

    def get_shapesensing_node_parameters(self):
        # get the parameter names
        param_name_aas = shape_sensing_needle.ShapeSensingNeedleNode.PARAM_AAS
        param_name_chs = shape_sensing_needle.ShapeSensingNeedleNode.PARAM_CHS
        param_name_len = shape_sensing_needle.ShapeSensingNeedleNode.PARAM_NEEDLELENGTH
        param_names    = [param_name_aas, param_name_chs, param_name_len]

        # create the clients
        cli_param = self.create_client(GetParameters, f"ShapeSensingNeedle/get_parameters")

        # get the parameter results
        req = GetParameters.Request(
            names=param_names
        )
        self.get_logger().info(f"Requesting parameters from {cli_param.srv_name}")
        future = cli_param.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res: GetParameters.Response = future.result()
        self.get_logger().info(f"Retrieved parameters from {cli_param.srv_name}")

        for param, pval in zip(param_names, res.values):
            pval: ParameterValue

            if param == param_name_aas:
                val = pval.integer_value
                self.num_aas = val

            elif param == param_name_chs:
                val = pval.integer_value
                self.num_chs = val

            elif param == param_name_len:
                val = pval.double_value
                self.needle_length = val

            else:
                raise ValueError(f"{param} is not either the AAs, CHs or needle length param names: {param_name_aas} & {param_name_chs} & {param_name_len}")

        # for

        # destroy the clients
        self.destroy_client(cli_param)

        # re-call the function if missing information
        if not self.configured:
            return self.get_shapesensing_node_parameters()

        self.get_logger().info(
            f"Configured benchmarker for needle with AAs: {self.num_aas} | CHs: {self.num_chs} | length: {self.needle_length} mm"
        )

        return {
            param_name_aas: self.num_aas,
            param_name_chs: self.num_chs,
            param_name_len: self.needle_length,
        }

    # get_shapesensing_node_parameters

    def publish_curvatures(self):
        if self.num_aas is None:
            return

        curvatures: np.ndarray = np.random.randn(2, self.num_aas).astype(np.float64)

        msg = Float64MultiArray(
            data=curvatures.ravel(order='F').tolist(),
            layout=MultiArrayLayout(
                dim=[
                    MultiArrayDimension(
                        label="x",
                        stride=curvatures.dtype.itemsize,
                        size=curvatures.shape[0],
                    ),
                    MultiArrayDimension(
                        label="y",
                        stride=curvatures.dtype.itemsize,
                        size=curvatures.shape[0],
                    ),
                ]
            )
        )

        self.pub_curvatures.publish(msg)

    # publish_curvatures

    def publish_entrypoint(self):
        msg = Point()

        msg.x = 0.0
        msg.y = 0.0
        msg.z = 0.0

        self.pub_entrypoint.publish(msg)

    # publish_entrypoint

    def publish_needlepose(self):
        msg = PoseStamped()

        msg.pose.position.z = float(self.current_insertion_depth)
        msg.header.stamp    = self.get_clock().now().to_msg()

        self.pub_needlepose.publish(msg)
        if self.current_insertion_depth_ts is None:
            self.current_insertion_depth_ts = self.get_clock().now().nanoseconds

    # publish_stage_state

    def save(self, odir: str = None):
        # filename config
        if odir is None:
            odir = self.odir
            
        ofile_base    = os.path.join(odir, "shape_sensing_needle_node_benchmark")
        ofile_results = ofile_base + "_results.xlsx"
        ofile_plots   = ofile_base + "_plots.png"

        # compute the time averages
        summary = self.summarize_results()

        # save the pandas data tables
        with pd.ExcelWriter(ofile_results, mode='w') as xl_writer:
            self.benchmark_data.to_excel(xl_writer, sheet_name="results")
            for lbl, tbl in summary.items():
                tbl: pd.DataFrame
                tbl.to_excel(xl_writer, sheet_name=lbl)

            # for
        # with
        self.get_logger().info(f"Saved benchmark statistics file tables to: {ofile_results}")

        # plot the results & save
        lenchange_ts: pd.DataFrame = summary["length_change_ts"]
        tbl_stats   : pd.DataFrame = summary["stats_ts"]
        labels                     = pd.unique(tbl_stats.columns.get_level_values(0))
        
        fig = plt.figure(figsize=(10, 10))
        axs = fig.subplots(nrows=len(labels) + 1, sharex=True, sharey=True)

        # plot the results
        for ax, lbl in zip(axs[:-1], labels):
            lbl: str
            ax: plt.Axes

            ax.errorbar(
                x=tbl_stats.index.to_numpy(),
                y=tbl_stats[(lbl, "mean")].to_numpy(),
                yerr=tbl_stats[[(lbl, "min"), (lbl, "max")]].to_numpy().transpose(),
                label=lbl.split("_")[1],
                linewidth=3,
            )
            ax.set_title(lbl.split("_")[1])
            ax.set_ylabel("Message Time Delay (ns)")

        # for
        
        axs[-1].plot(
            lenchange_ts.index.to_numpy(),
            lenchange_ts["delta_lenchange_timestamp__ns"].to_numpy(),
            label="Length Change",
            linewidth=4,
        )
        axs[-1].set_title("length change update")
        axs[-1].set_ylabel("Message Time Delay (ns)")

        fig.suptitle("Distribution of Time Delay for Shape-Sensing Publishing over Insertion Depths")
        axs[-1].set_xlabel("Insertion Depth (mm)")

        fig.savefig(ofile_plots)
        plt.close(fig)
        self.get_logger().info(f"Saved benchmark results plot to: {ofile_plots}")

    # save

    def sub_shape_callback(self, msg: PoseArray):
        # get the current and message timestamps
        ros_ts = self.get_clock().now().nanoseconds
        msg_ts = Time.from_msg(msg.header.stamp).nanoseconds

        self.benchmark_data = pd.concat(
            [
                self.benchmark_data,
                pd.DataFrame.from_dict(
                    {
                        "insertion_depth__mm"          : [self.current_insertion_depth],
                        "insertion_depth_timestamp__ns": [self.current_insertion_depth_ts],
                        "msg_timestamp__ns"            : [msg_ts],
                        "ros_timestamp__ns"            : [ros_ts],
                    },
                )
            ],
            axis=0,
            ignore_index=True,
        )

        # update the needle pose
        self.update_needlepose()

    # sub_shape_callback

    def summarize_results(self):
        # compute the time deltas
        self.compute_time_deltas()

        # perform statistical analysis over the results
        summary = dict()

        summary["length_change_ts"] = self.benchmark_data.groupby("insertion_depth__mm").mean()
        summary["stats_ts"]         = self.benchmark_data[[
            "insertion_depth__mm",
            "delta_msg_timestamp__ns",
            "delta_ros_timestamp__ns",
        ]].groupby("insertion_depth__mm").agg(
            [
                'min',
                'mean',
                'max',
            ]
        )

        return summary

    # summarize_results

    def update_needlepose(self):
        """ Update the needle pose to the next needle pose"""
        num_current_samples = np.count_nonzero(self.benchmark_data["insertion_depth__mm"] == self.current_insertion_depth)

        if num_current_samples < self.num_samples: # not enough samples yet
            return

        self.get_logger().info(f"Completed benchmark for insertion depth: {self.current_insertion_depth} mm")
        self.current_insertion_depth   += self.current_insertion_depth_inc
        self.current_insertion_depth_ts = None
        self.get_logger().info(f"Benchmarking insertion depth: {self.current_insertion_depth} mm")

        if self.current_insertion_depth > self.needle_length:
            return

        self.publish_needlepose()

    # update_needlepose

# class: ShapeSensingBenchmarker

def main(args=None):
    rclpy.init(args=args)

    
    # configure nodes
    ssbenchmarker_node = ShapeSensingBenchmarker()
    ssbenchmarker_node.get_shapesensing_node_parameters()

    try:

        while not ssbenchmarker_node.completed:
            rclpy.spin_once(ssbenchmarker_node)

        # while

    except KeyboardInterrupt:
        pass

    # clean-up
    ssbenchmarker_node.save()

    ssbenchmarker_node.destroy_node()

    rclpy.shutdown()

# main


if __name__ == "__main__":
    main()

# if __main__