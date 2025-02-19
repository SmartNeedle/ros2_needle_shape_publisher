import numpy as np
from typing import List
# ROS2 packages
import rclpy
from rclpy.parameter import Parameter
# messages
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

# custom package
from .sensorized_shape_sensing_needle import NeedleNode
from . import utilities


class SensorizedNeedleNode( NeedleNode ):
    """Needle to handle sensor calibration/curvatures"""
    # - FBG signal parameters
    # -- number of signals to gather
    PARAM_NUMSIGNALS = ".".join( [ NeedleNode.PARAM_NEEDLE, "sensor.numberSignals" ] )
    # -- temperature compensation
    PARAM_TEMPCOMP = ".".join( [ NeedleNode.PARAM_NEEDLE, "sensor.temperatureCompensate" ] )

    def __init__( self, name="SensorizedNeedle" ):
        super().__init__( name )

        # declare and get parameters
        pd_numsamples = ParameterDescriptor(
                name=self.PARAM_NUMSIGNALS, type=Parameter.Type.INTEGER.value )
        self.num_samples = self.declare_parameter(
                pd_numsamples.name, descriptor=pd_numsamples,
                value=200 ).get_parameter_value().integer_value

        pd_tempcomp = ParameterDescriptor(
                name=self.PARAM_TEMPCOMP, type=Parameter.Type.BOOL.value,
                description="Whether to perform temperature compensation or not.",
                )
        self.temperature_compensate = self.declare_parameter(
                pd_tempcomp.name, descriptor=pd_tempcomp, value=False
                ).get_parameter_value().bool_value
        
        if self.temperature_compensate:
            self.get_logger().info(f"Using temperature compensation!")

        else:
            self.get_logger().warn(f"Not using temperature compensation")

        # make all positive since we are using processed wavelengths
        self.ss_needle.ref_wavelengths = np.ones_like( self.ss_needle.ref_wavelengths )
        #
        # container of wavelengths
        self.__wavelength_container = np.zeros(
                (self.num_samples,
                 self.ss_needle.num_channels * self.ss_needle.num_activeAreas) )
        self.__wavelength_container_idx = 0  # the row index to place the sample
        self.__wavelength_container_full = False  # whether we have collected a full number of samples

        # create subscriptions
        self.sub_signals = self.create_subscription(
                Float64MultiArray, 'sensor/processed', self.sub_signals_callback,
                10 )
        # create publishers
        self.pub_curvatures = self.create_publisher( Float64MultiArray, 'state/curvatures', 10 )

        # create timers
        self.pub_curvatures_timer = self.create_timer( 0.01, self.publish_curvatures )

        # set parameter callback function
        self.add_on_set_parameters_callback( self.parameters_callback )

    # __init__

    def parameters_callback( self, parameters: List[ Parameter ] ):
        """ Parameter set calllbacks"""
        successful = True
        reasons = [ ]
        for param in parameters:
            if param.name == self.PARAM_NUMSIGNALS:
                num_samples = param.get_parameter_value().integer_value
                try:
                    self.get_logger().info( f"Updating number of signals to {num_samples}..." )
                    self.update_numsamples( num_samples )
                    self.get_logger().info( f"Updated number of signals to {num_samples}." )

                # try
                except ValueError as e:
                    successful = False
                    reasons.append( f"{self.PARAM_NUMSIGNALS} must be > 0" + '\n' + str( e ) )
                    self.get_logger().error(
                            "Update failed. Did not set the updated number of signals." )

                # except
            # if: parameter: NUMSIGNALS

            elif param.name == self.PARAM_TEMPCOMP:
                if param.type_ is Parameter.Type.BOOL:
                    self.temperature_compensate = param.get_parameter_value().bool_value

                else:
                    successful = False

            # elif: TEMPCOMP
            else:
                reasons.append( f"{param.name} is not a valid settable-parameter." )

        # for

        return SetParametersResult( successful=successful, reason="\n".join( reasons ) )

    # parameters_callback

    def publish_curvatures( self ):
        """ Publish the curvatures of the shape-sensing needle"""
        # current_curvatures are N x 2 ( columns are: x,  y ) -> ravel('F') -> (X_AA1, X_AA2, ..., Y_AA1, Y_AA2,...)
        curvatures = self.ss_needle.current_curvatures.ravel( order='F' )

        self.get_logger().debug(f"Curvatures: {self.ss_needle.current_curvatures}")
        itemsize = self.ss_needle.current_curvatures.dtype.itemsize
        dimx = MultiArrayDimension(
                label="x", stride=itemsize,
                size=self.ss_needle.current_curvatures.shape[ 0 ] * itemsize )
        dimy = MultiArrayDimension(
                label="y", stride=itemsize,
                size=self.ss_needle.current_curvatures.shape[ 0 ] * itemsize )

        msg = Float64MultiArray(
                data=curvatures.tolist(), 
                layout=MultiArrayLayout( dim=[ dimx, dimy ] ) )

        self.pub_curvatures.publish( msg )

    # publish_curvatures

    def sub_signals_callback( self, msg: Float64MultiArray ):
        """ Call back to processed signals subscription"""
        # get the FBG signals
        # TODO: perform appending by channel
        signals_dict = utilities.unpack_fbg_msg( msg )
        self.get_logger().debug( f"Signals dictionary unpacked: {signals_dict}" )
        # signals = np.array( list( signals_dict.values() ) ).ravel()  # to be improved
        signals = np.hstack( list( signals_dict.values() ) ).ravel()

        # perform temperature compensation
        if self.temperature_compensate:
            signals = self.ss_needle.temperature_compensate( signals, arg_check=True )

        # if

        self.get_logger().debug( f"Signals unwrapped: {signals}" )
        self.get_logger().debug(
                f"Shape of signals: {signals.shape} | Shape of wl container: {self.__wavelength_container.shape}" )

        # add the signals to the container
        if signals.shape[0] != self.__wavelength_container.shape[1]: # check if the correct size to avoid errors.
            self.get_logger().warn(f"Wrong sensor shape size! Size is {signals.shape}, needs size of {(self.__wavelength_container.shape[1],)}")
            return 
        # if 
        self.__wavelength_container[ self.__wavelength_container_idx ] = signals
        self.__wavelength_container_idx += 1
        if self.__wavelength_container_idx >= self.num_samples:
            self.__wavelength_container_full = True

        # if

        # update the wavelengths
        if self.__wavelength_container_full:
            self.__wavelength_container_idx %= self.num_samples
            self.ss_needle.update_wavelengths( self.__wavelength_container, processed=True )

        # if

    # sub_signals_callback

    def update_numsamples( self, num_samples: int ):
        """ Update the number of FBG samples """
        if num_samples <= 0:
            raise ValueError( "Number of samples must be > 0" )

        # if
        elif num_samples != self.num_samples:  # shrink the size
            # reshape the wavelength container
            if num_samples < self.num_samples:
                self.__wavelength_container = self.__wavelength_container[ :num_samples ]

            # if
            elif num_samples > self.num_samples:  # grow the size
                if self.__wavelength_container_full:
                    self.__wavelength_container = np.vstack(
                            (
                                    self.__wavelength_container[ self.__wavelength_container_idx: ],
                                    self.__wavelength_container[ :self.__wavelength_container_idx ]
                                    ) )  # reorder to be @ bottom
                    self.__wavelength_container_idx = self.num_samples - 1  # place pointer at the end

                # if

                # append to the wavelength container
                self.__wavelength_container = np.vstack(
                        (self.__wavelength_container, np.zeros(
                                (num_samples - self.num_samples,
                                 self.__wavelength_container.shape[ 1 ]) )) )

                self.__wavelength_container_full = False

            # elif

            # update number of samples and check if the wavelength container is full (mainly for shrinking)
            self.num_samples = num_samples
            if self.__wavelength_container_idx >= self.num_samples:
                self.__wavelength_container_full = True

            # if
            self.__wavelength_container_idx %= self.num_samples

        # elif

    # update_numsamples


# class: SensorizedNeedleNode

def main( args=None ):
    rclpy.init( args=args )

    sensorized_needle_node = SensorizedNeedleNode()

    try:
        rclpy.spin( sensorized_needle_node )

    except KeyboardInterrupt:
        pass

    # clean-up
    sensorized_needle_node.destroy_node()
    rclpy.shutdown()


# main

if __name__ == "__main__":
    main()

# if __main__
