#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import math

class TrajectoryEvaluator:
    def __init__(self):
        """
        Initialize publishers once when the class is instantiated
        """
        # Publishers for RViz visualization - initialized once
        self.observed_pub = rospy.Publisher('/trajectory_observed', MarkerArray, queue_size=10)
        self.gt_pub = rospy.Publisher('/trajectory_ground_truth', MarkerArray, queue_size=10)
        self.pred_pub = rospy.Publisher('/trajectory_predictions', MarkerArray, queue_size=10)
        
        print("TrajectoryEvaluator initialized with publishers")

    def calculate_metrics(self, observations, ground_truth, predictions):
        """
        Calculate ADE (Average Displacement Error) and FDE (Final Displacement Error)
        """
        if len(observations) == 0 or len(ground_truth) == 0 or len(predictions) == 0:
            return None, None, None, None
            
        obs = observations[0] if len(observations) > 0 else None
        gt = ground_truth[0] if len(ground_truth) > 0 else None
        pred = predictions
        
        if obs is None or gt is None or pred is None:
            return None, None, None, None
            
        # Convert to numpy arrays
        obs_np = np.array(obs)
        gt_np = np.array(gt)
        pred_np = np.array(pred)
        
        print(f"Observations shape: {obs_np.shape}")
        print(f"Ground Truth shape: {gt_np.shape}")
        print(f"Predictions shape: {pred_np.shape}")
        
        # Calculate ADE (Average Displacement Error) between predictions and ground truth
        if len(pred_np) <= len(gt_np):
            # Use the overlapping portion
            min_len = min(len(pred_np), len(gt_np))
            ade = np.mean(np.linalg.norm(pred_np[:min_len] - gt_np[:min_len], axis=1))
            # FDE (Final Displacement Error) - error at the last predicted point
            fde = np.linalg.norm(pred_np[min_len-1] - gt_np[min_len-1])
        else:
            # Predictions are longer than ground truth
            ade = np.mean(np.linalg.norm(pred_np[:len(gt_np)] - gt_np, axis=1))
            fde = np.linalg.norm(pred_np[len(gt_np)-1] - gt_np[-1])
        
        # Calculate prediction horizon metrics (how far ahead we're predicting)
        pred_horizon = len(pred_np)
        
        # Calculate total distance traveled in ground truth
        gt_distances = np.linalg.norm(np.diff(gt_np, axis=0), axis=1)
        total_distance = np.sum(gt_distances)
        
        return ade, fde, pred_horizon, total_distance

    def create_trajectory_markers(self, points, namespace, color, marker_id_offset=0):
        """
        Create MarkerArray for trajectory visualization
        """
        marker_array = MarkerArray()
        
        if points is None or len(points) == 0:
            return marker_array
            
        # Convert to numpy array if it's a list
        if isinstance(points, list):
            points = np.array(points[0]) if len(points) > 0 else np.array([])
        
        if len(points) == 0:
            return marker_array
            
        # Create line strip for trajectory
        line_marker = Marker()
        line_marker.header.frame_id = "map"  # Adjust frame_id as needed
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = f"{namespace}_line"
        line_marker.id = marker_id_offset
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.05  # Line width
        line_marker.color = color
        
        # Add points to line strip
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.1  # Slightly above ground
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # Create sphere markers for individual points
        for i, point in enumerate(points):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "map"
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.ns = f"{namespace}_points"
            sphere_marker.id = marker_id_offset + i + 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = float(point[0])
            sphere_marker.pose.position.y = float(point[1])
            sphere_marker.pose.position.z = 0.1
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = 0.1
            sphere_marker.scale.y = 0.1
            sphere_marker.scale.z = 0.1
            sphere_marker.color = color
            
            marker_array.markers.append(sphere_marker)
        
        return marker_array

    def publish_trajectories_to_rviz(self, observations, ground_truth, predictions):
        """
        Publish trajectory visualizations to RViz using pre-initialized publishers
        """
        # Define colors
        blue_color = ColorRGBA(0.0, 0.0, 1.0, 0.8)    # Blue for observations
        green_color = ColorRGBA(0.0, 1.0, 0.0, 0.8)   # Green for ground truth
        red_color = ColorRGBA(1.0, 0.0, 0.0, 0.8)     # Red for predictions
        
        # Create and publish observed trajectory
        obs_markers = self.create_trajectory_markers(observations, "observed", blue_color, 0)
        self.observed_pub.publish(obs_markers)
        
        # Create and publish ground truth trajectory
        gt_markers = self.create_trajectory_markers(ground_truth, "ground_truth", green_color, 100)
        self.gt_pub.publish(gt_markers)
        
        # Create and publish predicted trajectory
        pred_markers = self.create_trajectory_markers([predictions], "predictions", red_color, 200)
        self.pred_pub.publish(pred_markers)
        
        print("Published trajectory markers to RViz")

    def print_metrics(self, ade, fde, pred_horizon, total_distance):
        """
        Print formatted metrics results
        """
        print("\n" + "="*50)
        print("TRAJECTORY EVALUATION RESULTS")
        print("="*50)
        
        if ade is not None:
            print(f"Average Displacement Error (ADE): {ade:.4f} meters")
            print(f"Final Displacement Error (FDE): {fde:.4f} meters")
            print(f"Prediction Horizon: {pred_horizon} time steps")
            print(f"Total GT Distance: {total_distance:.4f} meters")
            if total_distance > 0:
                print(f"ADE as % of total distance: {(ade/total_distance)*100:.2f}%")
        else:
            print("Could not calculate metrics - invalid data")
        
        print("="*50)

    def evaluate_and_visualize(self, observations, ground_truth, predictions):
        """
        Complete function to calculate metrics and publish visualizations
        """
        # Calculate metrics
        ade, fde, pred_horizon, total_distance = self.calculate_metrics(observations, ground_truth, predictions)
        
        # Print metrics
        self.print_metrics(ade, fde, pred_horizon, total_distance)
        
        # Publish visualizations
        self.publish_trajectories_to_rviz(observations, ground_truth, predictions)
        
        return ade, fde

    def publish_trajectory(self, observations, ground_truth, predictions):
        """
        Main method to be called from real_time_simulation.py
        Combines evaluation and visualization
        """
        return self.evaluate_and_visualize(observations, ground_truth, predictions)

# Backward compatibility functions for standalone usage
def publish_trajectory(observations, ground_truth, predictions):
    """
    Function wrapper for backward compatibility
    Creates a temporary instance if needed
    """
    if not hasattr(publish_trajectory, 'evaluator'):
        publish_trajectory.evaluator = TrajectoryEvaluator()
    
    return publish_trajectory.evaluator.publish_trajectory(observations, ground_truth, predictions)
