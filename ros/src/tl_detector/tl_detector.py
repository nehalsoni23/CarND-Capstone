#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy import spatial
import tf, cv2, yaml, math
import numpy as np

STATE_COUNT_THRESHOLD = 3

GT_ENABLED = True

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.light = 0
        self.tree = None
        self.light_wp_list = None
        self.light_state_list = ["RED", "YELLOW", "GREEN"]

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.stopline_positions = self.config['stop_line_positions']

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.light_wp_list = self.get_stopline_wps()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
            rospy.loginfo('State of the Traffic light: {}'.format(self.light_state_list[self.state]))
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        car_x = pose.position.x
        car_y = pose.position.y
        car_w = pose.orientation.w

        theta = 2 * np.arccos(car_w)

        if theta > np.pi:
            theta = -(2 * np.pi - theta)

        if self.tree == None:
            waypoint_list = []
            for index, waypoint in enumerate(self.waypoints.waypoints):
                x = waypoint.pose.pose.position.x
                y = waypoint.pose.pose.position.y
                waypoint_list.append([x, y])
            # Loading waypoints to KD Tree for searching
            self.tree = spatial.KDTree(np.asarray(waypoint_list))

        # TODO_NEHAL - Check for car_x replacement to self.pose.position.x
        distance, index = self.tree.query([car_x, car_y])

        next_wp_x = self.waypoints.waypoints[index].pose.pose.position.x
        next_wp_y = self.waypoints.waypoints[index].pose.pose.position.y

        head = np.arctan2((next_wp_y - car_y),(next_wp_x - car_x))
        angle = abs(theta - head)

        if angle > (np.pi / 4):
            index = (index + 1) % len(self.waypoints.waypoints)
        return index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Since I'm using ground truth and not implementing TLClassifier,
        # so directly returning the light's state
        return self.lights[light].state
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # List of positions that correspond to the line to stop in front of traffic light for a given intersection
        if(self.pose and self.light_wp_list):
            closest_wp = self.get_closest_waypoint(self.pose.pose)
            light_wp = self.light_wp_list[self.light]["waypoint"]

            # TODO find the closest visible traffic light (if one exists)
            if (closest_wp > light_wp):
                # TODO_NEHAL
                # max_light_wp = self.light_wp_list[-1]["waypoint"]
                # if(self.light != 0 or closest_wp < max_light_wp):
                self.light = (self.light + 1) % len(self.light_wp_list)
                light_wp = self.light_wp_list[self.light]["waypoint"]

            state = self.get_light_state(self.light)
            return light_wp, state
        else:
            return -1, TrafficLight.UNKNOWN

    def get_stopline_wps(self):
        stop_lines_wps_list = []

        for i in range(len(self.stopline_positions)):
            stop_line = Pose()
            stop_line.position.x = self.stopline_positions[i][0]
            stop_line.position.y = self.stopline_positions[i][1]

            closest_light_wp = self.get_closest_waypoint(stop_line)

            wp_dict = {'index':i, 'waypoint':closest_light_wp}

            stop_lines_wps_list.append(wp_dict)

        sorted_wps_list = sorted(stop_lines_wps_list, key=lambda k: k['waypoint'])
        return sorted_wps_list

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
