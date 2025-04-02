import cv2, os, time
import torch

torch.set_num_threads(1)  # Reduce CPU threads

if torch.cuda.is_available():
    torch.cuda.device("cpu")
    
import numpy as np
import mediapipe as mp
import time
import json
from datetime import datetime

from constant import *
from supabase import create_client, Client



import warnings
warnings.filterwarnings("ignore")


SUPABASE_URL = "https://icnnnxyoppcauqpaiulw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imljbm5ueHlvcHBjYXVxcGFpdWx3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMwNDE1OTMsImV4cCI6MjA1ODYxNzU5M30.9ZoP7LZNq-gIaC73RTEleBabNJf3f78dh8NMlsy6gdE"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load YOLO model (YOLOv5)
def get_ball_positions(filename):

    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.3  # Confidence threshold

    # Load the video
    video_path = UPLOAD_FOLDER+'/'+filename
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    #temp_video = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4")

    output_path = OUTPUT_FOLDER+"/"+filename

    # output_path = "outputs/"+filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read the first frame for motion detection
    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (1, 1), 0)

    # Create a background subtractor for better motion detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
    padding = 15 
    player_boxes = []
    ball_positions = []
    DEPTH_SCALING_FACTOR = 1000  

    peak = False
    previous_coordinate = None
    end_tracking  = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends
        
        left_limit = int(frame_width * 0.15)  # 15% from left
        right_limit = int(frame_width * 0.85)  # 85% from left
        lower_limit = int(frame_height * 0.80)  


        overlay = frame.copy()

    # Define red color for exclusion zones (BGR format: Blue, Green, Red)
        red_color = (0, 0, 255)  # Pure red

        # Draw left padding area
        cv2.rectangle(overlay, (0, 0), (left_limit, frame_height), red_color, -1)

        # Draw right padding area
        cv2.rectangle(overlay, (right_limit, 0), (frame_width, frame_height), red_color, -1)

        # Blend the overlay with the original frame using addWeighted
        alpha = 0.4  # Transparency factor (0 = fully transparent, 1 = fully opaque)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


        frame_height, frame_width, _ = frame.shape 

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (1, 1), 0)

        # Apply motion detection (Background subtraction + Frame Difference)
        fg_mask = bg_subtractor.apply(gray)
        frame_diff = cv2.absdiff(prev_frame, gray)
        
        # Combine motion masks
        combined_mask = cv2.bitwise_and(fg_mask, frame_diff)

        # Apply threshold to refine detection
        _, thresh = cv2.threshold(combined_mask, 3, 255, cv2.THRESH_BINARY)

        # Morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours (moving objects)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # YOLO Detection for players
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Extract bounding boxes

        largest_player = None
        max_area = 0
        
        for *xyxy, conf, cls in detections:
            if int(cls) == 0:  # Assuming class 0 is 'person' in YOLOv5
                x1, y1, x2, y2 = map(int, xyxy)

                x1 = max(x1 - padding, 0)
                y1 = max(y1 - padding, 0)
                x2 = min(x2 + padding, frame_width)
                y2 = min(y2 + padding, frame_height)

                area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area

        # Check if this is the largest detected player
                if area > max_area:
                    max_area = area
                    largest_player = (x1, y1, x2, y2)

                player_boxes.append((x1, y1, x2, y2))

            if int(cls) == 32:
              
              # Draw rectangle around detected player
              cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
              # # Add label
              label = f"Ball: {conf:.2f}"
              cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # if largest_player:
        #         x1, y1, x2, y2 = largest_player
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box for largest player
        #         cv2.putText(frame, "Largest Player", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)  # Get perimeter (True = closed shape)

                # Avoid division by zero
                if perimeter == 0:
                    continue

                circularity = (4 * np.pi * area) / (perimeter ** 2)  # Circularity formula

                ball_center = (x + w // 2, y + h // 2)  # Get ball center point

                ball_radius = (w + h) / 4  

                # **Skip balls outside the center vertical strip**
                if ball_center[0] < left_limit or ball_center[0] > right_limit or ball_center[1] > lower_limit:
                    continue

                if end_tracking:
                    continue

                ball_inside_player = False
                for (px1, py1, px2, py2) in player_boxes:
                    # px1 = largest_player[0]
                    # py1 = largest_player[1]
                    # px2 = largest_player[2]
                    # py2 = largest_player[3]
                        if px1 <= x <= px2 and py1 <= y <= py2:
                            ball_inside_player = True
                            break

                # if not ball_inside_player:
                  # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                  # cv2.putText(frame, f"Ball: {circularity:.2f} -Area{area} {x,y}", (x-100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if the detected object is ball-like
                if 10.5 < area < 1500 and 0.5 < w/h < 1.5 and circularity > 0.76:  # More circularity = closer to round
                    
                    # ball_inside_player = False
                    if not ball_inside_player:

                        Z = DEPTH_SCALING_FACTOR / (ball_radius + 1e-6)
                        # Draw rectangle around detected object
                        if not previous_coordinate:
                          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                          cv2.putText(frame, f"Ball: {circularity:.2f} {x,y,Z}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                          ball_positions.append((x,y,Z))
                          previous_coordinate = (x,y,Z)
                        elif not peak and previous_coordinate[1] < y:
                          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                          cv2.putText(frame, f"Ball: {circularity:.2f} {x,y,Z}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                          ball_positions.append((x,y,Z))
                          previous_coordinate = (x,y,Z)
                        elif peak and previous_coordinate[1] > y:
                          end_tracking = True
                        else:
                          peak = True
        

        #ball_positions_new = filter_projectile_trajectory(ball_positions)

        if len(ball_positions) > 1:
                for i in range(1, len(ball_positions)):
                    cv2.line(frame, (ball_positions[i - 1][0], ball_positions[i - 1][1]), (ball_positions[i][0],ball_positions[i][1]), (0, 255, 255), 2)

        # Update previous frame
        prev_frame = gray.copy()

        # Write the frame into the file
        out.write(frame)

        # Show tracking result
        #cv2.imshow("Ball and Player Tracking", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    response = ""
    with open(output_path, "rb") as video_file:
        unique_filename = f"{int(time.time())}_{filename}"
        response = supabase.storage.from_("outputs").upload(f"processed_videos/{unique_filename}", video_file)

    print("Uploaded to Supabase:", response)

    # Clean up: Delete the temporary video file
    os.remove(video_path)
    os.remove(output_path)


    print(f"Tracking video saved to {output_path}")
    # print(ball_positions)
    # return ball_positions
    return supabase.storage.from_("outputs").get_public_url(response.path)

class CricketBatsmanAnalysis:
    def __init__(self, input_video_path, output_video_path, use_mediapipe=True):
        """
        Initialize the cricket batsman analysis system.
        
        Args:
            input_video_path (str): Path to the input video file
            output_video_path (str): Path where processed video will be saved
            use_mediapipe (bool): Use MediaPipe (True) or OpenPose (False)
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.use_mediapipe = use_mediapipe
        self.trajectory_data = []
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video at {input_video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input video properties:")
        print(f"  - FPS: {self.fps}")
        print(f"  - Frame size: {self.frame_width}x{self.frame_height}")
        print(f"  - Total frames: {self.total_frames}")
        print(f"  - Duration: {self.total_frames/self.fps:.2f} seconds")
        
        # Initialize MediaPipe Pose
        if self.use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # Initialize OpenPose (Comment out if not using OpenPose)
            try:
                from openpose import pyopenpose as op
                params = dict()
                params["model_folder"] = "models/"
                params["model_pose"] = "BODY_25"
                params["net_resolution"] = "-1x368"
                
                self.opWrapper = op.WrapperPython()
                self.opWrapper.configure(params)
                self.opWrapper.start()
                self.datum = op.Datum()
            except ImportError:
                raise ImportError("OpenPose not installed. Please install it or use MediaPipe instead.")
        
        # Initialize video writer with explicit fourcc for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 output
        self.output_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.output_writer.isOpened():
            raise ValueError(f"Failed to open output video writer for {output_video_path}")
        
        # Define key joint mapping for consistent output structure
        self.joint_mapping = {
            'nose': 0,
            'neck': 1,  # For MediaPipe, we'll approximate this
            'right_shoulder': 12,
            'right_elbow': 14,
            'right_wrist': 16,
            'left_shoulder': 11,
            'left_elbow': 13,
            'left_wrist': 15,
            'right_hip': 24,
            'right_knee': 26,
            'right_ankle': 28,
            'left_hip': 23,
            'left_knee': 25,
            'left_ankle': 27
        }

        # Add confidence threshold for valid joint detection
        self.confidence_threshold = 0.5  # Adjust this threshold as needed

    def process_video(self):
        """Process the input video and produce the output with pose overlay"""
        frame_count = 0
        start_time = time.time()
        
        # Reset the video capture to start from the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process each frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of video reached after {frame_count} frames")
                break
            
            frame_count += 1
            
            # Process frame with pose detection
            if self.use_mediapipe:
                processed_frame, joint_data = self._process_with_mediapipe(frame, frame_count)
            else:
                processed_frame, joint_data = self._process_with_openpose(frame, frame_count)
            
            # Add frame data to trajectory
            self.trajectory_data.append({
                "frame": frame_count,
                "joints": joint_data
            })
            
            # Log coordinates to console
            self._log_coordinates(frame_count, joint_data)
            
            # Write processed frame to output video
            self.output_writer.write(processed_frame)
            
            # Display progress
            if frame_count % 10 == 0 or frame_count == 1:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processing frame {frame_count}/{self.total_frames} ({frame_count/self.total_frames*100:.1f}%), FPS: {fps:.2f}")
        
        # Check if we processed the expected number of frames
        if frame_count < self.total_frames:
            print(f"WARNING: Only processed {frame_count} frames out of {self.total_frames}")
            print(f"Processed duration: {frame_count/self.fps:.2f} seconds")
        
        # Release resources
        self.cap.release()
        
        # Ensure all frames are written and close the output writer
        self.output_writer.release()
        
        # Save trajectory data to JSON file
        #self._save_trajectory_data()
        
        total_elapsed = time.time() - start_time
        print(f"Processing complete. Total time: {total_elapsed:.2f} seconds")
        print(f"Output saved to {self.output_video_path}")
        
        # Verify the output video duration
        output_cap = cv2.VideoCapture(self.output_video_path)
        if output_cap.isOpened():
            output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = output_cap.get(cv2.CAP_PROP_FPS)
            output_duration = output_frames / output_fps if output_fps > 0 else 0
            output_cap.release()
            print(f"Output video properties:")
            print(f"  - FPS: {output_fps}")
            print(f"  - Frames: {output_frames}")
            print(f"  - Duration: {output_duration:.2f} seconds")
        else:
            print("Could not open output video to verify duration")
        
        return self.trajectory_data
    
    def _process_with_mediapipe(self, frame, frame_count):
        """Process frame using MediaPipe Pose"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(frame_rgb)
        
        joint_data = {}
        
        if results.pose_landmarks:
            # Extract 3D coordinates
            for joint_name, landmark_idx in self.joint_mapping.items():
                # Handle neck approximation (average of shoulders)
                if joint_name == 'neck':
                    left_shoulder = results.pose_landmarks.landmark[self.joint_mapping['left_shoulder']]
                    right_shoulder = results.pose_landmarks.landmark[self.joint_mapping['right_shoulder']]
                    
                    # Check if both shoulders are detected with sufficient confidence
                    if left_shoulder.visibility >= self.confidence_threshold and right_shoulder.visibility >= self.confidence_threshold:
                        x = (left_shoulder.x + right_shoulder.x) / 2
                        y = (left_shoulder.y + right_shoulder.y) / 2
                        z = (left_shoulder.z + right_shoulder.z) / 2
                        visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
                        
                        # Store normalized coordinates with depth (z)
                        joint_data[joint_name] = (float(x), float(y), float(z))
                    else:
                        joint_data[joint_name] = None
                else:
                    landmark = results.pose_landmarks.landmark[landmark_idx]
                    
                    # Check if the landmark is detected with sufficient confidence
                    if landmark.visibility >= self.confidence_threshold:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        
                        # Store normalized coordinates with depth (z)
                        joint_data[joint_name] = (float(x), float(y), float(z))
                    else:
                        # Mark as None if the landmark is not detected with sufficient confidence
                        joint_data[joint_name] = None
            
            # Draw the pose
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If no pose is detected, set all joints to None
            for joint_name in self.joint_mapping.keys():
                joint_data[joint_name] = None
            
            # Add frame counter even if no pose is detected
            cv2.putText(frame, f"Frame: {frame_count} (No pose detected)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, joint_data
    
    def _process_with_openpose(self, frame, frame_count):
        """Process frame using OpenPose"""
        # Set image data to process with OpenPose
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        
        joint_data = {}
        
        if self.datum.poseKeypoints is not None and len(self.datum.poseKeypoints) > 0:
            # Extract keypoints from the first person detected
            keypoints = self.datum.poseKeypoints[0]
            
            # OpenPose BODY_25 model keypoints
            openpose_mapping = {
                'nose': 0,
                'neck': 1,
                'right_shoulder': 2,
                'right_elbow': 3,
                'right_wrist': 4,
                'left_shoulder': 5,
                'left_elbow': 6,
                'left_wrist': 7,
                'right_hip': 9,
                'right_knee': 10,
                'right_ankle': 11,
                'left_hip': 12,
                'left_knee': 13,
                'left_ankle': 14
            }
            
            # Extract joint data
            for joint_name, idx in openpose_mapping.items():
                if idx < len(keypoints):
                    x, y, confidence = keypoints[idx]
                    
                    # Check if the point is detected with sufficient confidence
                    if confidence >= self.confidence_threshold * 100:  # OpenPose confidence is 0-100
                        # Normalize coordinates
                        norm_x = x / self.frame_width
                        norm_y = y / self.frame_height
                        
                        # OpenPose doesn't provide z directly, so we use confidence as an approximation
                        # or set z to 0 for 2D analysis
                        z = 0.0  # Or use confidence / 100.0 for a rough approximation
                        
                        # Store normalized coordinates with pseudo-depth
                        joint_data[joint_name] = (float(norm_x), float(norm_y), float(z))
                    else:
                        # Mark as None if the point is not detected with sufficient confidence
                        joint_data[joint_name] = None
                else:
                    # Mark as None if the joint index is out of bounds
                    joint_data[joint_name] = None
            
            # Use OpenPose's rendered output
            frame = self.datum.cvOutputData
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If no pose is detected, set all joints to None
            for joint_name in self.joint_mapping.keys():
                joint_data[joint_name] = None
            
            # Add frame counter even if no pose is detected
            cv2.putText(frame, f"Frame: {frame_count} (No pose detected)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, joint_data
    
    def _log_coordinates(self, frame_count, joint_data):
        """Log 3D joint coordinates to console"""
        if frame_count % 30 == 0:  # Log every 30 frames to avoid console spam
            print(f"\nFrame {frame_count} Joint Coordinates:")
            for joint_name, coords in joint_data.items():
                if coords is not None:
                    print(f"  {joint_name}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")
                else:
                    print(f"  {joint_name}: None")
    
    def _save_trajectory_data(self):
        """Save trajectory data to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_data_{timestamp}.json"
        
        # Create a serializable version of the data
        serializable_data = []
        for frame_data in self.trajectory_data:
            serializable_frame = {
                "frame": frame_data["frame"],
                "joints": {}
            }
            
            for joint_name, coords in frame_data["joints"].items():
                serializable_frame["joints"][joint_name] = coords if coords is not None else None
            
            serializable_data.append(serializable_frame)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Trajectory data saved to {filename}")
        
        # Return the list for direct use
        return self.trajectory_data
    


def process_player(filename):
    # Example usage

    input_video = UPLOAD_FOLDER+'/'+filename

    output_video = OUTPUT_FOLDER+"/"+filename

    # input_video = "videos/player3.mp4"  # Replace with your input video path
    # output_video = "outputs/player3.mp4"
    
    # Choose between MediaPipe (True) or OpenPose (False)
    use_mediapipe = True
    
    try:
        analyzer = CricketBatsmanAnalysis(input_video, output_video, use_mediapipe)
        trajectory_data = analyzer.process_video()
        
        # Example: accessing the data
        print(f"\nProcessed {len(trajectory_data)} frames")
        if len(trajectory_data) > 0:
            print(f"Sample joint data from first frame:")
            for joint, coords in trajectory_data[0]['joints'].items():
                if coords is not None:
                    print(f"  {joint}: {coords}")
                else:
                    print(f"  {joint}: None")
        
        response = ""
        with open(output_video, "rb") as video_file:
            unique_filename = f"{int(time.time())}_{filename}"
            response = supabase.storage.from_("outputs").upload(f"processed_videos/{unique_filename}", video_file)

        print("Uploaded to Supabase:", response)

        # Clean up: Delete the temporary video file
        os.remove(input_video)
        os.remove(output_video)


        print(f"Tracking video saved to {output_video}")
        # print(ball_positions)
        # return ball_positions
        return supabase.storage.from_("outputs").get_public_url(response.path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()