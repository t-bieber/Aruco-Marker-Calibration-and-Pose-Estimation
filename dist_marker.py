import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

# === Configuration ===
reference_markers = {
    20: np.array([0.0, 167.0]),       # (x, y) in mm
    21: np.array([290.0, 167.0]),
    22: np.array([0.0, 0.0]),
    23: np.array([290.0, 0.0]),
}
tracked_marker_id = 0

# Optional names for readability
marker_names = {
    0: "Actor",
    20: "Top-Left Ref",
    21: "Top-Right Ref",
    22: "Bottom-Left Ref",
    23: "Bottom-Right Ref"
}

# === Load camera calibration ===
if not os.path.exists('./calibration/CameraCalibration.pckl'):
    print("You need to calibrate the camera you'll be using.")
    exit()
else:
    with open('./calibration/CameraCalibration.pckl', 'rb') as f:
        (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration data is invalid. Recalibrate.")
        exit()
    else:
        print("Camera matrix:\n", cameraMatrix)
        print("Distortion coefficients:\n", distCoeffs)

# === ArUco Setup ===
ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.GridBoard(size=(1, 1), markerLength=0.09, markerSeparation=0.02, dictionary=ARUCO_DICT)

# === Camera stream setup ===
cv2.namedWindow('ProjectImage', cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(3, 1280)
cam.set(4, 720)

while cam.isOpened():
    ret, ProjectImage = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(ProjectImage, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    corners, ids, rejectedImgPoints, _ = aruco.refineDetectedMarkers(
        image=gray, board=board,
        detectedCorners=corners, detectedIds=ids,
        rejectedCorners=rejectedImgPoints,
        cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, borderColor=(0, 0, 255))

    if ids is not None and len(ids) > 0:
        rotation_vectors, translation_vectors, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.05, cameraMatrix, distCoeffs)

        marker_poses = {}
        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec = rotation_vectors[i], translation_vectors[i]
            marker_poses[int(marker_id)] = (rvec, tvec)
            ProjectImage = cv2.drawFrameAxes(ProjectImage, cameraMatrix, distCoeffs, rvec, tvec, 0.025)

        # Draw names of reference markers
        for ref_id, _ in reference_markers.items():
            if ref_id in marker_poses:
                marker_index = np.where(ids.flatten() == ref_id)[0][0]
                marker_corners = corners[marker_index][0]
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1])) - 10
                label = marker_names.get(ref_id, f"ID {ref_id}")
                cv2.putText(ProjectImage, label, (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw name for the tracked object as well
        if tracked_marker_id in marker_poses:
            marker_index = np.where(ids.flatten() == tracked_marker_id)[0][0]
            marker_corners = corners[marker_index][0]
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1])) - 10
            tracked_label = marker_names.get(tracked_marker_id, f"Tracked Object")
            cv2.putText(ProjectImage, tracked_label, (center_x - 40, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)


        if tracked_marker_id in marker_poses:
            tracked_rvec, tracked_tvec = marker_poses[tracked_marker_id]
            tracked_R, _ = cv2.Rodrigues(tracked_rvec)
            T_tracked = np.eye(4)
            T_tracked[:3, :3] = tracked_R
            T_tracked[:3, 3] = tracked_tvec.flatten()

            poses_mm = []
            angles_deg = []

            print(f"\n=== Tracking Marker {tracked_marker_id} ({marker_names.get(tracked_marker_id, '')}) ===")

            # Iterate over all reference markers to calculate pose
            for ref_id, ref_pos_mm in reference_markers.items():
                if ref_id not in marker_poses:
                    continue
                ref_rvec, ref_tvec = marker_poses[ref_id]
                ref_R, _ = cv2.Rodrigues(ref_rvec)
                T_ref = np.eye(4)
                T_ref[:3, :3] = ref_R
                T_ref[:3, 3] = ref_tvec.flatten()

                T_rel = np.linalg.inv(T_ref) @ T_tracked
                rel_translation = T_rel[:3, 3]  # in meters
                rel_pos_mm = ref_pos_mm + rel_translation[:2] * 1000
                poses_mm.append(rel_pos_mm)

                R_relative = T_rel[:3, :3]
                yaw_rad = np.arctan2(R_relative[1, 0], R_relative[0, 0])
                yaw_deg = np.degrees(yaw_rad)
                yaw_deg_clockwise = (-yaw_deg) % 360
                angles_deg.append(yaw_deg_clockwise)

                # Print the data for each reference
                print(f"→ From Reference {ref_id} ({marker_names.get(ref_id, '')}):")
                print(f"   X: {rel_pos_mm[0]:.1f} mm, Y: {rel_pos_mm[1]:.1f} mm, Z: {rel_translation[2]*1000:.1f} mm, Rot: {yaw_deg_clockwise:.1f}°")

            # Compute average pose
            if poses_mm:
                poses_mm = np.array(poses_mm)
                avg_position = np.mean(poses_mm, axis=0)
                avg_rotation = np.mean(angles_deg)

                # Draw on video
                marker_index = np.where(ids.flatten() == tracked_marker_id)[0][0]
                marker_corners = corners[marker_index][0]
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1])) - 10

                text_pos = f"X: {avg_position[0]:.0f} mm, Y: {avg_position[1]:.0f} mm"
                text_rot = f"Rot: {avg_rotation:.1f}°"
                cv2.putText(ProjectImage, text_pos, (center_x - 60, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(ProjectImage, text_rot, (center_x - 60, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                print(f"→ Averaged:")
                print(f"   X: {avg_position[0]:.1f} mm, Y: {avg_position[1]:.1f} mm, Rot: {avg_rotation:.1f}°")

    cv2.imshow('ProjectImage', ProjectImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
