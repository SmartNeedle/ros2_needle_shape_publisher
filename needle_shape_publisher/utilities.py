import numpy as np

# ROS messages
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header

# custom packages
from needle_shape_sensing import geometry


def calculate_needle_length(shape: np.ndarray):
    return np.sum(
        np.linalg.norm(
            np.diff(shape, n=1, axis=0),
            ord=2, 
            axis=1
        )
    )

# calculate_needle_length

def msg2pose( msg: Pose ) -> np.ndarray:
    """ Convert a Pose message into a pose"""
    pos = np.array( [ msg.position.x, msg.position.y, msg.position.z ] )
    quat = np.array( [ msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z ] )
    R = geometry.quat2rotm( quat )

    return pos, R


# msg2pose

def msg2poses( msg: PoseArray ) -> np.ndarray:
    """ Convert a pose message into an array of 4x4 matrices (N, 4, 4) of rigid poses"""
    poses = list()

    for pose_msg in msg.poses:
        tf = np.eye(4)
        
        p, R = msg2pose(pose_msg)
        tf[:3, :3] = R
        tf[:3, 3] = p

        poses.append(tf)

    # for
    
    return np.stack(poses, axis=0)

def pose2msg( pos: np.ndarray, R: np.ndarray ) -> Pose:
    """ Turn a pose into a Pose message """
    msg = Pose()

    # handle position
    msg.position.x = pos[ 0 ]
    msg.position.y = pos[ 1 ]
    msg.position.z = pos[ 2 ]

    # handle orientation
    quat = geometry.rotm2quat( R )
    msg.orientation.w = quat[ 0 ]
    msg.orientation.x = quat[ 1 ]
    msg.orientation.y = quat[ 2 ]
    msg.orientation.z = quat[ 3 ]

    return msg


# pose2msg

def poses2msg( pmat: np.ndarray, Rmat: np.ndarray, header: Header = Header() ):
    """ Turn a sequence of poses into a PoseArray message"""
    # determine number of elements in poses
    N = min( pmat.shape[ 0 ], Rmat.shape[ 0 ] )

    # generate the message and add the individual poses
    msg = PoseArray( header=header )
    for i in range( N ):
        msg.poses.append( pose2msg( pmat[ i ], Rmat[ i ] ) )

    # for

    return msg


# poses2msg

def unpack_fbg_msg( msg ) -> dict:
    """ Unpack Float64MultiArray into dict of numpy arrays """
    ret_val = { }
    idx_i = 0

    for dim in msg.layout.dim:
        ch_num = int( dim.label.strip( 'CH' ) )
        size = int( dim.size / dim.stride )

        ret_val[ ch_num ] = np.float64( msg.data[ idx_i:idx_i + size ] )

        idx_i += size  # increment size to next counter

    # for

    return ret_val

# unpack_fbg_msg