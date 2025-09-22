import rospy
import json
import os
from human_awareness_msgs.msg import PersonsList
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from code.eval.evaluator import Evaluator
from code.utils.process_frames import process_frames

class RealTimeDataCollector:
    def __init__(self):
        self.image_detections_msg = []
        self.local_frames = []
        self.ego_frames = []
        self.valid_local_frames_count = 0
        self.latest_tf = None
        self.seq_len = 10
        self.interval = 5 # our detection frequency is 5Hz, so interval of 5 means 1 second

        rospy.init_node('real_time_data_collector', anonymous=True)
        
        # Initialize the evaluator once
        self.evaluator = Evaluator()
        
        # Subscribers
        self.image_detections_sub = rospy.Subscriber('/image_detections', PersonsList, self.image_detections_callback)
        self.tf_sub = rospy.Subscriber('/robot_tf', TransformStamped, self.tf_callback)
        
        # Publisher for trajectory visualization
        self.trajectory_pub = rospy.Publisher('/trajectory_predictions', MarkerArray, queue_size=10)

    def tf_callback(self, msg):
        self.latest_tf = msg

    def image_detections_callback(self, msg):
        if not msg.persons:
            return
        rospy.logdebug(f"Received PersonsList with {len(msg.persons)} persons.")
        self.image_detections_msg.append(msg)
        
        # First create ego frame to ensure TF data
        if not self.create_ego_frame():
            rospy.logwarn("Skipping frame due to missing TF data")
            return
            
        self.create_local_frame(msg)
        renumbered_local_frames, renumbered_ego_frames = self.renumber_frames()
        #self.save_frames(renumbered_local_frames, renumbered_ego_frames)

        X, Y, name, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path = process_frames(self.seq_len, self.interval, renumbered_local_frames, renumbered_ego_frames)

        predictions = self.evaluator.evaluate_traj_pred(X, Y, name, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path)

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

    def create_ego_frame(self):
        # Get latest tf data from the subscriber
        if self.latest_tf is not None:
            coordinates = {
                "x": self.latest_tf.transform.translation.x,
                "y": self.latest_tf.transform.translation.z,  # Swap y and z
                "z": self.latest_tf.transform.translation.y,
                "q1": self.latest_tf.transform.rotation.x,
                "q2": self.latest_tf.transform.rotation.y,
                "q3": self.latest_tf.transform.rotation.z,
                "q4": self.latest_tf.transform.rotation.w
            }
            frame_data = {
                "frame": self.valid_local_frames_count,
                "coordinates": coordinates
            }
            self.ego_frames.append(frame_data)
            return True
        else:
            rospy.logwarn_throttle(1, "No TF data received yet")
            exit()
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

    def save_frames(self, local_frame_data, ego_frame_data):
        outdir = os.path.join('code', 'real_time_data', 'files')
        os.makedirs(outdir, exist_ok=True)
        local_file_path = os.path.join(outdir, 'local_frames.json')
        with open(local_file_path, 'a') as f:
            for frame_data in local_frame_data:
                f.write(json.dumps(frame_data) + '\n')
        ego_file_path = os.path.join(outdir, 'ego_frames.json')
        with open(ego_file_path, 'a') as f:
            for frame_data in ego_frame_data:
                f.write(json.dumps(frame_data) + '\n')

if __name__ == '__main__':
    collector = RealTimeDataCollector()
    collector.spin()
