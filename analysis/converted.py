#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import Quaternion
from scipy.signal import lfilter
import numpy as np
from tf.transformations import euler_from_quaternion
import math

# Initialize ROS node
rospy.init_node('magnetometer_calibration')

# Constants
Fs = 40  # Sampling frequency in Hz

# Callback function for IMU data
def imu_callback(msg):
    # Extract linear acceleration and orientation
    accel_x = msg.linear_acceleration.x
    accel_y = msg.linear_acceleration.y
    accel_z = msg.linear_acceleration.z

    orientation = msg.orientation
    orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

    # Process IMU data
    # Insert your IMU data processing logic here

# Callback function for GPS data
def gps_callback(msg):
    # Extract GPS data
    latitude = msg.latitude
    longitude = msg.longitude
    altitude = msg.altitude

    # Process GPS data
    # Insert your GPS data processing logic here

# Subscribe to IMU and GPS topics
rospy.Subscriber('/imu', Imu, imu_callback)
rospy.Subscriber('/gps', NavSatFix, gps_callback)

# Filter parameters for the complementary filter
alpha_mag = 0.98
alpha_imu = 0.02

# Initialize variables for the complementary filter
yaw_filtered = 0.0
lpf_y = 0.0
hpf_x = 0.0

# ROS message loop
rate = rospy.Rate(Fs)

while not rospy.is_shutdown():
    # Perform complementary filter calculations
    lpf_y = alpha_mag * lpf_y + (1 - alpha_mag) * yaw
    hpf_x = alpha_imu * (hpf_x + yaw - last_yaw)

    yaw_filtered = lpf_y + hpf_x

    # Publish filtered yaw angle or perform additional processing

    # Store previous yaw for high-pass filter
    last_yaw = yaw

    rate.sleep()

# Finish the ROS node
rospy.spin()
