import cv2
import mediapipe as mp
import numpy as np
import yaml
from dataclasses import dataclass
from typing import Optional
from collections import deque
import time

with open('config.yaml') as f:
    CFG = yaml.safe_load(f)

T = CFG['thresholds']


@dataclass
class FaceMetrics:
    ear_left:       float = 0.0
    ear_right:      float = 0.0
    ear:            float = 0.0
    mar:            float = 0.20
    gaze_x:         float = 0.0
    gaze_y:         float = 0.0
    head_pitch:     float = 0.0
    head_yaw:       float = 0.0
    head_roll:      float = 0.0
    blink_detected: bool  = False
    yawn_detected:  bool  = False
    eyes_closed:    bool  = False
    face_visible:   bool  = True


class FaceTracker:

    LEFT_EYE   = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
    LEFT_IRIS  = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    MOUTH      = [61, 291, 39, 181, 0, 17, 269, 405]
    POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]

    POSE_3D = np.array([
        [ 0.0,    0.0,    0.0  ],
        [ 0.0,  -330.0, -65.0 ],
        [-225.0,  170.0,-135.0],
        [ 225.0,  170.0,-135.0],
        [-150.0, -150.0,-125.0],
        [ 150.0, -150.0,-125.0],
    ], dtype=np.float64)

    EAR_BLINK  = T['ear_blink']
    EAR_CLOSED = T['ear_closed']
    MAR_YAWN   = T['mar_yawn']

    def __init__(self):
        self.mp_face   = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self._prev_ear   = 1.0
        self._closed_ctr = 0
        self._blink_ts   = deque()

    def _eye_aspect_ratio(self, lm, indices, w, h):
        pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in indices],
            dtype=np.float32
        )
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return float((A + B) / (2.0 * C + 1e-6))

    def _mouth_aspect_ratio(self, lm, w, h):
        pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in self.MOUTH],
            dtype=np.float32
        )
        vertical   = (np.linalg.norm(pts[2] - pts[6]) +
                      np.linalg.norm(pts[3] - pts[7])) / 2.0
        horizontal = np.linalg.norm(pts[0] - pts[1])
        return float(vertical / (horizontal + 1e-6))

    def _gaze_ratio(self, lm, eye_idx, iris_idx, w, h):
        eye_pts  = np.array([[lm[i].x*w, lm[i].y*h] for i in eye_idx],  dtype=np.float32)
        iris_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in iris_idx], dtype=np.float32)
        eye_cx,  eye_cy  = eye_pts.mean(axis=0)
        iris_cx, iris_cy = iris_pts.mean(axis=0)
        eye_w = np.linalg.norm(eye_pts[0] - eye_pts[3]) + 1e-6
        eye_h = np.linalg.norm(eye_pts[1] - eye_pts[5]) + 1e-6
        return (float((iris_cx - eye_cx) / eye_w),
                float((iris_cy - eye_cy) / eye_h))

    def _head_pose(self, lm, w, h):
        img_pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in self.POSE_LANDMARKS],
            dtype=np.float64
        )
        focal = w
        cam   = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
        dist  = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            self.POSE_3D, img_pts, cam, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])

    def process(self, frame) -> Optional[FaceMetrics]:
        h_px, w_px = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return FaceMetrics(face_visible=False)

        lm = result.multi_face_landmarks[0].landmark

        ear_l = self._eye_aspect_ratio(lm, self.LEFT_EYE,  w_px, h_px)
        ear_r = self._eye_aspect_ratio(lm, self.RIGHT_EYE, w_px, h_px)
        ear   = (ear_l + ear_r) / 2.0

        blink = (ear < self.EAR_BLINK) and (self._prev_ear >= self.EAR_BLINK)
        if blink:
            self._blink_ts.append(time.time())

        if ear < self.EAR_CLOSED:
            self._closed_ctr += 1
        else:
            self._closed_ctr  = 0
        eyes_closed  = self._closed_ctr > 15

        self._prev_ear = ear

        gx_l, gy_l = self._gaze_ratio(lm, self.LEFT_EYE,  self.LEFT_IRIS,  w_px, h_px)
        gx_r, gy_r = self._gaze_ratio(lm, self.RIGHT_EYE, self.RIGHT_IRIS, w_px, h_px)
        gaze_x = (gx_l + gx_r) / 2.0
        gaze_y = (gy_l + gy_r) / 2.0

        mar = self._mouth_aspect_ratio(lm, w_px, h_px)
        pitch, yaw, roll = self._head_pose(lm, w_px, h_px)

        return FaceMetrics(
            ear_left=ear_l, ear_right=ear_r, ear=ear,
            mar=mar,
            gaze_x=gaze_x, gaze_y=gaze_y,
            head_pitch=pitch, head_yaw=yaw, head_roll=roll,
            blink_detected=blink,
            yawn_detected=(mar > self.MAR_YAWN),
            eyes_closed=eyes_closed,
            face_visible=True
        )

    def blinks_per_minute(self):
        cutoff = time.time() - 60.0
        while self._blink_ts and self._blink_ts[0] < cutoff:
            self._blink_ts.popleft()
        return float(len(self._blink_ts))

    def draw_overlay(self, frame, metrics):
        if not metrics.face_visible:
            cv2.putText(frame, 'NO FACE DETECTED', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return
        color = (0, 255, 0) if metrics.ear > self.EAR_BLINK else (0, 0, 255)
        cv2.putText(frame, f'EAR: {metrics.ear:.3f}',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f'Gaze: ({metrics.gaze_x:.2f}, {metrics.gaze_y:.2f})',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(frame,
                    f'Yaw:{metrics.head_yaw:.1f} Pitch:{metrics.head_pitch:.1f}',
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        if metrics.blink_detected:
            cv2.putText(frame, 'BLINK!',
                        (frame.shape[1] - 130, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        if metrics.yawn_detected:
            cv2.putText(frame, 'YAWN!',
                        (frame.shape[1] - 130, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)