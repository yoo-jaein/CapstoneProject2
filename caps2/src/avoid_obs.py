#!/usr/bin/env python

# knu capstion project 2
# avoid_obs.py
# testing in world

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def callback(dt):
    
    thr = 0.8 # Laser scan range threshold

    if dt.ranges[0]>thr and dt.ranges[15]>thr and dt.ranges[345]>thr: # Checks if there are obstacles in front and 15 degrees left and right
								      # Can change threshold and angle value

        move.linear.x = 0.5 # go forward (linear velocity)
        move.angular.z = 0.0 # do not rotate (angular velocity)
    else:
        move.linear.x = 0.0 # stop the robot
        move.angular.z = 0.5 # rotate counter-clockwise

        if dt.ranges[0]>thr and dt.ranges[15]>thr and dt.ranges[345]>thr:
            move.linear.x = 0.5
            move.angular.z = 0.0
    pub.publish(move) # publish the move object


move = Twist() # Creates a Twist message type object
rospy.init_node('avoid_obs_node') # Initializes a node

pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)  # Publisher object which will publish "Twist" type messages
                            				 # on the "/cmd_vel" Topic, "queue_size" is the size of the
                                                         # outgoing message queue used for asynchronous publishing

sub = rospy.Subscriber("/scan", LaserScan, callback)  # Subscriber object which will listen "LaserScan" type messages
                                                      # from the "/scan" Topic and call the "callback" function
						      # each time it reads something from the Topic

rospy.spin() # Loops infinitely until someone stops the program execution

