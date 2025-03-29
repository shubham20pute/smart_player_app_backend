import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
from constant import *

from util import *

app = Flask(__name__)

CORS(app)


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/player_motion", methods=["POST"])
def player_motion():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    file.save(UPLOAD_FOLDER+'/'+file.filename)


    response = process_player(file.filename)
    print(response)

    return jsonify({"processed_video_url": response}), 200



@app.route("/process_video", methods=["POST"])
def ballTracking():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    file.save(UPLOAD_FOLDER+'/'+file.filename)


    response = get_ball_positions(file.filename)
    print(response)

    return jsonify({"processed_video_url": response}), 200

# @app.route("/process_video", methods=["POST"])
# def process_video():
#     if "video" not in request.files:
#         print('no file found')
#         return jsonify({"error": "No video file provided"}), 400
    
#     video_file = request.files["video"]
#     filename = secure_filename(video_file.filename)
#     input_path = os.path.join(UPLOAD_FOLDER, filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
    
#     video_file.save(input_path)
    
#     # HSV color range for detecting a red ball
#     lower_red1 = np.array([0, 120, 70])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 120, 70])
#     upper_red2 = np.array([180, 255, 255])
    
#     contrail_color = (255, 0, 0)  # Red contrail
#     track_color = (0, 255, 0)  # Green track line

#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         return jsonify({"error": "Couldn't open the video file"}), 500
    
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
#     pts = []  # For contrail effect
#     track_points = []  # For continuous tracking
#     total_distance = 0  # Distance traveled

#     def calculate_distance(pt1, pt2):
#         return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
#         mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
#         mask_red = cv2.bitwise_or(mask_red1, mask_red2)
#         kernel = np.ones((3, 3), np.uint8)
#         mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
#         mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
#         cnts, _ = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         center = None
        
#         if cnts:
#             c = max(cnts, key=cv2.contourArea)
#             ((x, y), radius) = cv2.minEnclosingCircle(c)
#             center = (int(x), int(y))
#             pts.append(center)
#             track_points.append(center)
#             if len(pts) > 15:
#                 pts.pop(0)
            
#             cv2.circle(frame, center, int(radius), contrail_color, 2)
#             cv2.circle(frame, center, 5, (0, 255, 255), -1)
            
#             for i in range(1, len(pts)):
#                 if pts[i - 1] and pts[i]:
#                     thickness = int(np.sqrt(len(pts) - i) * 2.5)
#                     cv2.line(frame, pts[i - 1], pts[i], contrail_color, thickness)
            
#             if len(track_points) > 1:
#                 total_distance += calculate_distance(track_points[-2], track_points[-1])
#             cv2.putText(frame, f"Distance: {int(total_distance)} px", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
#         for i in range(1, len(track_points)):
#             if track_points[i - 1] and track_points[i]:
#                 cv2.line(frame, track_points[i - 1], track_points[i], track_color, 2)
        
#         out.write(frame)
    
#     cap.release()
#     out.release()
#     #cv2.destroyAllWindows()

#     base_url = request.host_url  # This will return something like "http://192.168.1.100:5000/"

#     # Construct the full URL
#     full_url = f"{base_url}download/{os.path.basename(output_path)}"

#     return jsonify({"output_url": full_url})
    


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype="video/mp4")
    #return send_from_directory('static/outputs', filename, mimetype="video/mp4")





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
