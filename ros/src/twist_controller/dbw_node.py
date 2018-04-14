#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class CarParams(object):
    def __init__(self):
        self.vehicle_mass = None
        self.fuel_capacity = None
        self.brake_db = None
        self.decel_limit = None
        self.accel_limit = None
        self.wheel_radius = None
        self.wheel_base = None
        self.steer_ratio = None
        self.max_lat_accel = None
        self.max_steer_angle = None
        self.min_speed = None

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        car_params = CarParams()
        
        car_params.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        car_params.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        car_params.brake_deadband = rospy.get_param('~brake_deadband', .1)
        car_params.decel_limit = rospy.get_param('~decel_limit', -5)
        car_params.accel_limit = rospy.get_param('~accel_limit', 1.)
        car_params.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        car_params.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        car_params.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        car_params.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        car_params.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # TODO: Create `Controller` object
        self.twist_ctlr = Controller(car_params=car_params)
        
        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.vel_cb, queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb, queue_size=1)

        # Values which needs to be monitored
        self.ref_lin_vel = 0
        self.ref_ang_vel = 0
        self.lin_vel = 0
        self.ang_vel = 0
        self.dbw_enabled = False
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            throttle, brake, steering = self.twist_ctlr.control(current_time, self.ref_lin_vel,
                                                                self.ref_ang_vel, self.lin_vel,
                                                                self.ang_vel, self.dbw_enabled)
            if self.dbw_enabled:
                self.publish(throttle, brake, steer)
                
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data
        rospy.loginfo('dbw is enabled: {}'.format(self.dbw_enabled))

    def vel_cb(self, msg):
        self.lin_vel = msg.twist.linear.x
        self.ang_vel = msg.twist.angular.z

    def twist_cmd_cb(self, msg):
        self.ref_lin_vel = msg.twist.linear.x
        self.ref_ang_vel = msg.twist.angular.z

if __name__ == '__main__':
    DBWNode()
