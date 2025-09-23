import rospy
import json
import os
from human_awareness_msgs.msg import PersonsList
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_msgs.msg import TFMessage
from code.eval.evaluator import Evaluator
from code.utils.process_frames import process_frames
from view_predictions import TrajectoryEvaluator

class RealTimeDataCollector:
    def __init__(self):
        self.image_detections_msg = []
        self.local_frames = []
        self.ego_frames = []
        self.camera_frames = []
        self.valid_local_frames_count = 0
        self.latest_robot_tf = None
        self.latest_camera_tf = None
        self.seq_len = 10
        self.interval = 5 # our detection frequency is 5Hz, so interval of 5 means 1 second
        self.debug = True

        rospy.init_node('real_time_data_collector', anonymous=True)
        
        # Initialize the evaluator and trajectory evaluator once
        self.evaluator = Evaluator()
        self.trajectory_evaluator = TrajectoryEvaluator()
        
        # Subscribers
        self.image_detections_sub = rospy.Subscriber('/image_detections', PersonsList, self.image_detections_callback)
        self.tf_sub = rospy.Subscriber('/combined_tf', TFMessage, self.tf_callback)
        
        # Publisher for trajectory visualization
        self.trajectory_pub = rospy.Publisher('/trajectory_predictions', MarkerArray, queue_size=10)

    def tf_callback(self, msg):
        # Extract robot and camera transforms from TFMessage
        for i, transform in enumerate(msg.transforms):
            if transform.child_frame_id == "base_footprint":
                self.latest_robot_tf = transform
            elif transform.child_frame_id == "l_eye_link" or "camera" in transform.child_frame_id.lower():
                self.latest_camera_tf = transform

    def image_detections_callback(self, msg):
        if not msg.persons:
            return
        self.image_detections_msg.append(msg)
        
        # Create both ego and camera frames
        ego_created = self.create_ego_frame()
        camera_created = self.create_camera_pose_frame()
        
        # Skip frame only if both are missing
        if not ego_created and not camera_created:
            print("Skipping frame due to missing BOTH TF data")
            return
        elif not ego_created:
            print("Warning: Missing robot TF data, but continuing with camera data")
        elif not camera_created:
            print("Warning: Missing camera TF data, but continuing with robot data")
            
        self.create_local_frame(msg)
        renumbered_local_frames, renumbered_ego_frames = self.renumber_frames()

        if len(self.local_frames) > self.seq_len * self.interval + 1:
            X, Y, name, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path = process_frames(self.seq_len, self.interval, renumbered_local_frames, renumbered_ego_frames, self.camera_frames)
            self.save_json(X, Y, name, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path)
            observations, ground_truth, predictions = self.evaluator.evaluate_traj_pred(self.debug, X, Y, name, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path)
            #self.trajectory_evaluator.publish_trajectory(observations, ground_truth, predictions)


    def spin(self):
        rospy.loginfo('RealTimeDataCollector spinning...')
        rospy.spin()

    def create_local_frame(self, msg):
        CORRECT_ORDER = [
            "Nose", "LEye", "REye", "LEar", "REar",
            "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
            "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
        ]
        coordinates = []
        if msg.persons:
            person = msg.persons[0]
            # Map part_id to (x, y, confidence)
            part_dict = {part.part_id: (part.x, part.y, getattr(part, 'confidence', 1.0)) for part in person.body_parts}
            keypoints_flat = []
            for part_name in CORRECT_ORDER:
                if part_name in part_dict:
                    keypoints_flat.extend(part_dict[part_name])
                else:
                    print(f"Warning: {part_name} not found in body parts for person {person.id}. Using default values.")
                    exit(1)
            bbox = self.create_bbox(person.body_parts)
            coordinates.append({
                'id': 1,
                'x': getattr(person.body_pose.position, 'x', 0.0),
                'y': getattr(person.body_pose.position, 'z', 0.0),  # Swap y and z
                'z': getattr(person.body_pose.position, 'y', 0.0),
                'bbox': bbox,
                'keypoints': keypoints_flat
            })
        else:
            return None
        frame_data = {
            'frame': self.valid_local_frames_count,
            'coordinates': coordinates
        }
        self.local_frames.append(frame_data)

    def create_camera_pose_frame(self):
        # Get latest camera_pose tf data from the subscriber
        if self.latest_camera_tf is not None:
            coordinates = {
                "x": self.latest_camera_tf.transform.translation.x,
                "y": self.latest_camera_tf.transform.translation.z,  # Swap y and z
                "z": self.latest_camera_tf.transform.translation.y,
                "q1": self.latest_camera_tf.transform.rotation.x,
                "q2": self.latest_camera_tf.transform.rotation.y,
                "q3": self.latest_camera_tf.transform.rotation.z,
                "q4": self.latest_camera_tf.transform.rotation.w
            }
            frame_data = {
                "frame": self.valid_local_frames_count,
                "coordinates": coordinates
            }
            self.camera_frames.append(frame_data)
            return True
        else:
            print("No Camera_pose TF data available")
            rospy.logwarn_throttle(1, "No Camera_pose TF data received yet")
            return False
        
    def create_ego_frame(self):
        # Get latest robot tf data from the subscriber
        if self.latest_robot_tf is not None:
            coordinates = {
                "x": self.latest_robot_tf.transform.translation.x,
                "y": self.latest_robot_tf.transform.translation.z,  # Swap y and z
                "z": self.latest_robot_tf.transform.translation.y,
                "q1": self.latest_robot_tf.transform.rotation.x,
                "q2": self.latest_robot_tf.transform.rotation.y,
                "q3": self.latest_robot_tf.transform.rotation.z,
                "q4": self.latest_robot_tf.transform.rotation.w
            }
            frame_data = {
                "frame": self.valid_local_frames_count,
                "coordinates": coordinates
            }
            self.ego_frames.append(frame_data)
            return True
        else:
            print("No robot TF data available")
            rospy.logwarn_throttle(1, "No TF data received yet")
            return False

    def create_bbox(self, body_parts):
        min_x = min(part.x for part in body_parts)
        min_y = min(part.y for part in body_parts)
        max_x = max(part.x for part in body_parts)
        max_y = max(part.y for part in body_parts)
        return [min_x, min_y, max_x, max_y]


    def renumber_frames(self):
        number_frames = self.seq_len * self.interval + 1

        print(f"Frames: {self.valid_local_frames_count}/{number_frames} (seq_len={self.seq_len}, interval={self.interval})")
        self.valid_local_frames_count += 1
        
        # Initialize return variables
        renumbered_local_frames = []
        renumbered_ego_frames = []
        
        if len(self.local_frames) >= number_frames:
            # Get the first number_frames frames in chronological order and renumber them starting from 0
            local_frames_subset = self.local_frames[:number_frames]
            ego_frames_subset = self.ego_frames[:number_frames]
            
            # Renumber frames starting from 0, frames from the detection dont have a numbered id
            for i, frame in enumerate(local_frames_subset):
                renumbered_frame = frame.copy()
                renumbered_frame['frame'] = i
                renumbered_local_frames.append(renumbered_frame)
            
            for i, frame in enumerate(ego_frames_subset):
                renumbered_frame = frame.copy()
                renumbered_frame['frame'] = i
                renumbered_ego_frames.append(renumbered_frame)
        
        return renumbered_local_frames, renumbered_ego_frames

    def save_json(self, X, Y, name, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path):
        """
        Save the processed data in the same format as the training data JSON files
        """
        # Create the data structure matching the expected format
        data = {
            "X": X,
            "Y": Y,
            "names": name,
            "kps": kps,
            "boxes_3d": boxes_3d,
            "boxes_2d": boxes_2d,
            "K": K,
            "ego_pose": ego_pose,
            "camera_pose": camera_pose,
            "traj_3d_ego": traj_3d_ego,
            "image_path": image_path,
            "clst_ls": ["20"] * len(X) if X else []  # Default cluster labels
        }
        
        # Create output directory if it doesn't exist
        output_dir = "code/real_time_data/test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        import time
        timestamp = int(time.time())
        filename = f"realtime_data_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved real-time data to: {filepath}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")


if __name__ == '__main__':
    collector = RealTimeDataCollector()
    collector.spin()
