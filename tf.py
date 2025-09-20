#!/usr/bin/env python2
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def main():
    rospy.init_node('tf_publisher')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_pub = rospy.Publisher('/robot_tf', TransformStamped, queue_size=10)
    rate = rospy.Rate(50)  # 50 Hz

    rospy.loginfo("TF publisher node started")

    while not rospy.is_shutdown():
        try:
            # Look up the transform from base_footprint to odom (robot's position in world)
            # This gives you the robot's pose relative to the odometry frame
            trans = tf_buffer.lookup_transform('odom', 'base_footprint', rospy.Time(0), rospy.Duration(0.1))

            # Publish the transform
            tf_pub.publish(trans)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logdebug("TF lookup error: %s" % str(e))
        except Exception as e:
            rospy.logerr("Unexpected error: %s" % str(e))
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass