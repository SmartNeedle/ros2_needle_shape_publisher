import sys, os, tempfile
import json
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

pkg_needle_shape_publisher = get_package_share_directory('needle_shape_publisher')

def generate_launch_description():
    ld = LaunchDescription()

    # determine #chs and numAAs
    needleParamFile = "3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json"
    optimMaxIterations = 15

    arg_odir        = DeclareLaunchArgument("odir", default_value=tempfile.gettempdir())
    arg_insdepthinc = DeclareLaunchArgument("insertion_depth_increment", default_value="10.0")
    arg_numsamples  = DeclareLaunchArgument("num_samples", default_value="100")
    
    node_benchmarker = Node(
        package='needle_shape_publisher',
        namespace='needle',
        executable='shapesensing_benchmarker',
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                "benchmarker.odir"                     : LaunchConfiguration("odir"),
                "benchmarker.insertion_depth.increment": LaunchConfiguration("insertion_depth_increment"),
                "benchmarker.num_samples"              : LaunchConfiguration("num_samples"),
            }
        ]
        
    )
    node_ssneedle = Node(
        package='needle_shape_publisher',
        namespace='needle',
        executable='shapesensing_needle',
        output='screen',
        emulate_tty=True,
        parameters=[ 
            {
                'needle.paramFile'               : PathJoinSubstitution([pkg_needle_shape_publisher, 'needle_data', needleParamFile]),
                'needle.optimizer.max_iterations': TextSubstitution(text=str(optimMaxIterations)),
            } 
        ]
    )

    # launch description
    ld.add_action(arg_odir)
    ld.add_action(arg_insdepthinc)
    ld.add_action(arg_numsamples)

    ld.add_action(node_ssneedle)
    ld.add_action(node_benchmarker)

    return ld

# generate_launch_descrtiption