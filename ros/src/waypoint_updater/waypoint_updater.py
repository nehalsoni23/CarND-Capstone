#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy import spatial
import numpy as np
import math
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.act_velocity = None
        self.break_range = []
        self.no_brake = False
        
        self.base_waypoints = None
        self.next_waypoint = None
        self.last_stopline_waypoint = -1
        self.waypoint_velocity = self.kph_to_mps(rospy.get_param('/waypoint_loader/velocity'))

        self.tree = None

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

    def kph_to_mps(self, vel):
        return (vel * 5 / 18)

    def loop(self):
        if(self.pose is not None) and (self.base_waypoints is not None):
            self.next_waypoint = self.next_wp_idx()
            self.publish_waypoints(self.next_waypoint)

    def next_wp_idx(self):
        car_x = self.pose.position.x
        car_y = self.pose.position.y
        car_w = self.pose.orientation.w

        theta = 2 * np.arccos(car_w)

        if theta > np.pi:
            theta = -(2 * np.pi - theta)

        if self.tree == None:
            waypoint_list = []
            for index, waypoint in enumerate(self.base_waypoints):
                x = waypoint.pose.pose.position.x
                y = waypoint.pose.pose.position.y
                waypoint_list.append([x, y])
            # Loading waypoints to KD Tree for searching
            self.tree = spatial.KDTree(np.asarray(waypoint_list))

        # TODO_NEHAL - Check for car_x replacement to self.pose.position.x
        distance, index = self.tree.query([car_x, car_y])

        next_wp_x = self.base_waypoints[index].pose.pose.position.x
        next_wp_y = self.base_waypoints[index].pose.pose.position.y

        head = np.arctan2((next_wp_y - car_y),(next_wp_x - car_x))
        angle = abs(theta - head)

        if angle > (np.pi / 4):
            index = (index + 1) % len(self.base_waypoints)

        return index

    def publish_waypoints(self, next_waypoint):
        lane = Lane()
        lane.waypoints = []
        index = next_waypoint
        for i in range(LOOKAHEAD_WPS):
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = self.base_waypoints[index].pose.pose.position.x
            waypoint.pose.pose.position.y = self.base_waypoints[index].pose.pose.position.y
            waypoint.twist.twist.linear.x = self.base_waypoints[index].twist.twist.linear.x
            lane.waypoints.append(waypoint)
            index = (index + 1) % len(self.base_waypoints)

        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if (self.base_waypoints is not None and self.act_velocity != None):
            wp_idx = msg.data

            # Check for RED light
            # if there is RED light, apply brake if needed
            if(wp_idx != -1 and self.on_brake == False):
                dist = self.distance(self.base_waypoints, self.next_waypoint, wp_idx)
                idx_dist = wp_idx - self.next_waypoint
                s_per_idx = dist/idx_dist if idx_dist else 0.9
                
                ref_accel = (self.act_velocity**2)(2*(dist-4))

                if(ref_accel >= 2 and ref_accel < 10):

                    for i in range(self.next_waypoint, wp_idx +1):
                        updated_velocity = self.act_velocity**2 - 2*ref_accel*s_per_idx*(i-self.next_waypoint+1)
                        updated_velocity = np.sqrt(updated_velocity) if (updated_velocity >= 0.0) else 0
                        self.set_waypoint_velocity(self.next_waypoint, wp_idx)

                    self.brake_range.append((self.next_waypoint, wp_idx))
                    self.on_brake = True
                # TODO_NEHAL
                # elif (ref_accel < 2):
                #    pass
                else:
                    pass
            if(wp_idx == -1 and self.last_stopline_waypoint != -1):
                for i in self.brake_range:
                    for j in range(i[0], i[1]+1):
                        self.set_waypoint_velocity(self.base_waypoints, j, self.waypoint_velocity)
                    self.brake_range.remove(i)
                    self.on_brake = False
                    
            self.last_stopline_waypoint = wp_idx

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def velocity_cb(self, msg):
        self.act_velocity = msg.twist.linear.x

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
