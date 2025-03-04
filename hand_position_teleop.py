# Import required libraries
import argparse
import sys
import time
import crtk
import dvrk
import numpy
import PyKDL
from pynput import keyboard  # For keyboard input

import tf_transformations  # For quaternion conversion

class teleop_application:
    def __init__(self, ral, config_dict):
        self.ral = ral
        self.expected_interval = config_dict['expected_interval']
        self.arm1_name = config_dict['arm1_name']
        self.step_size = config_dict['step_size']
        self.joint_step_size = config_dict['joint_step_size']
        self.control_type = config_dict['control_type']
        self.arm1 = dvrk.psm(ral, self.arm1_name)
        self.running = False

        self.initial_hand_position = None  # Store initial hand position
        self.initial_hand_orientation = None  # Store initial hand orientation (quaternion)
        self.scaling_factor = 0.1  # Adjust for workspace scaling


    # Homing the arms
    def home(self):
        self.arm1.check_connections()
        # self.arm2.check_connections()
        print('Starting enable')
        # if not self.arm1.enable(10) or not self.arm2.enable(10):
        if not self.arm1.enable(10):
            sys.exit('Failed to enable within 10 seconds')
        print('Starting home')
        # if not self.arm1.home(10) or not self.arm2.home(10):
        if not self.arm1.home(10):
            sys.exit('Failed to home within 10 seconds')
        print('Homing complete')

    # Move the arm's tip incrementally in Cartesian coordinates
    def move_cartesian(self):

        current_pose1 = self.arm1.setpoint_cp() # type ='PyKDL.Frame'
        goal1 = PyKDL.Frame()
        goal1.p = current_pose1.p
        goal1.M = current_pose1.M
        goal1.p[0] += self.dx1 * self.step_size
        goal1.p[1] += self.dy1 * self.step_size
        goal1.p[2] += self.dz1 * self.step_size

        if self.control_type == 's':
            self.arm1.servo_cp(goal1)
        elif self.control_type == 'm':
            self.arm1.move_cp(goal1).wait()

        # current_pose2 = self.arm2.setpoint_cp()
        # goal2 = PyKDL.Frame()
        # goal2.p = current_pose2.p
        # goal2.M = current_pose2.M
        # goal2.p[0] += self.dx2 * self.step_size
        # goal2.p[1] += self.dy2 * self.step_size
        # goal2.p[2] += self.dz2 * self.step_size

        # if self.control_type == 's':
        #     self.arm2.servo_cp(goal2)
        # elif self.control_type == 'm':
        #     self.arm2.move_cp(goal2).wait()

    def move_joint(self):
        # Move joints
        current_joint_positions = self.arm1.setpoint_jp()
        current_jaw_position = self.arm1.jaw.setpoint_jp()

        new_joint_positions = current_joint_positions + self.joint_deltas
        new_jaw_position = current_jaw_position + self.jaw_delta

        self.arm1.servo_jp(new_joint_positions)
        self.arm1.jaw.servo_jp(new_jaw_position)

    def update_from_hand_pose(self, current_hand_position, current_hand_orientation):
        """
        Computes relative translation and rotation from initial hand pose and applies it to the dVRK end-effector.
        
        :param current_hand_position: [x, y, z] position of the hand.
        :param current_hand_orientation: [qx, qy, qz, qw] quaternion representing hand orientation.
        """
        if self.initial_hand_position is None:
            # Store the initial hand pose
            self.initial_hand_position = current_hand_position
            self.initial_hand_orientation = current_hand_orientation
            return

        # Compute relative translation
        hand_translation = [
            current_hand_position[i] - self.initial_hand_position[i]
            for i in range(3)
        ]

        # Scale the translation for the dVRK workspace
        scaled_translation = [
            hand_translation[i] * self.scaling_factor for i in range(3)
        ]

        # Convert initial and current orientations from quaternion to rotation matrices
        initial_rot_matrix = tf_transformations.quaternion_matrix(self.initial_hand_orientation)[:3, :3]
        current_rot_matrix = tf_transformations.quaternion_matrix(current_hand_orientation)[:3, :3]

        # Compute relative rotation: R_relative = R_current * R_initial_inv
        initial_rot_inv = tf_transformations.inverse_matrix(tf_transformations.quaternion_matrix(self.initial_hand_orientation))[:3, :3]
        relative_rot_matrix = current_rot_matrix @ initial_rot_inv  # Matrix multiplication

        # Convert relative rotation matrix to PyKDL format
        kdl_rotation = PyKDL.Rotation(
            relative_rot_matrix[0, 0], relative_rot_matrix[0, 1], relative_rot_matrix[0, 2],
            relative_rot_matrix[1, 0], relative_rot_matrix[1, 1], relative_rot_matrix[1, 2],
            relative_rot_matrix[2, 0], relative_rot_matrix[2, 1], relative_rot_matrix[2, 2]
        )

        # Get current dVRK end-effector pose
        current_pose = self.arm1.setpoint_cp()
        
        # Create goal pose by applying translation and rotation
        goal = PyKDL.Frame()
        goal.p = current_pose.p
        goal.M = current_pose.M

        # Apply translation
        goal.p[0] += scaled_translation[0]  # X-axis
        goal.p[1] += scaled_translation[1]  # Y-axis
        goal.p[2] += scaled_translation[2]  # Z-axis

        # Apply relative rotation
        goal.M = kdl_rotation * goal.M  # Apply rotation transformation

        # Move the dVRK end-effector
        if self.control_type == 's':
            self.arm1.servo_cp(goal)
        elif self.control_type == 'm':
            self.arm1.move_cp(goal).wait()

    def get_hand_pose():
        """
        Replace this function with actual hand position and orientation retrieval.
        It should return a list [x, y, z] for position and [qx, qy, qz, qw] for orientation.
        """
        mocap_position = [0.2, -0.1, 0.05]  # Example position
        mocap_orientation = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion (no rotation)
        return mocap_position, mocap_orientation


    def run(self):
        self.home()
        self.running = True

        while self.running:
            # Replace with actual mocap data retrieval
            current_hand_position, current_hand_orientation = get_hand_pose()

            # Apply pose update
            self.update_from_hand_pose(current_hand_position, current_hand_orientation)

            time.sleep(self.expected_interval)

if __name__ == '__main__':
    # Extract ROS arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:])  # Skip argv[0], script name

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a1', '--arm1', type=str, default='PSM1',
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help='arm1 name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    # parser.add_argument('-a2', '--arm2', type=str, default='PSM2',
    #                     choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
    #                     help='arm2 name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help='expected interval in seconds between messages sent by the device')
    parser.add_argument('-s', '--step_size', type=float, default=0.001,
                        help='step size for movement in meters')
    parser.add_argument('-j', '--joint_step_size', type=float, default=0.005,
                        help='step size for movement in meters')
    parser.add_argument('-c', '--control_type', type=str, default='s',
                        help='s - servo_cp, m - move_cp, i - interpolate_cp')
    args = parser.parse_args(argv)

    config_dict = {'arm1_name':args.arm1,
                #    'arm2_name':args.arm2,
                   'expected_interval':args.interval,
                   'step_size':args.step_size,
                   'joint_step_size':args.joint_step_size,
                   'control_type':args.control_type}

    ral = crtk.ral('teleop_keyboard')
    application = teleop_application(ral, config_dict)
    ral.spin_and_execute(application.run)