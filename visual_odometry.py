#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import cv2
import numpy as np
import os.path
import sys

from vision_tasks_base import NUM_FEATURES

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PinholeCamera:
    def __init__(self,
                 width=1241.0, height=376.0, # specialized for KITTI dataset
                 fx=718.8560, fy=718.8560,
                 cx=607.1928, cy=185.2157,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VisualOdometry:
    def __init__(self, vt, cam, dataset_path):
        self.frame_stage = 0
        self.vt = vt
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.kps = None
        self.desc = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.SIFT_create(nfeatures=NUM_FEATURES)
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        self.dataset_path = dataset_path
        self.annotations = self.loadAnnotations()

    def loadAnnotations(self):
        path = os.path.join(self.dataset_path, 'poses', '00.txt')
        with open(path) as f:
            return f.readlines()

    def getAbsoluteScale(self, frame_id): # specialized for KITTI dataset
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

    def processFirstFrame(self):
        self.px_ref, self.desc = self.detector.detectAndCompute(
                self.new_frame, None)
        self.kps = self.px_ref
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.vt.featureTracking(
                self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(
                self.px_cur, self.px_ref, focal=self.focal, pp=self.pp,
                method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(
                E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.vt.featureTracking(
                self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(
                self.px_cur, self.px_ref, focal=self.focal, pp=self.pp,
                method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(
                E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if self.px_ref.shape[0] < NUM_FEATURES:
            self.px_cur, self.desc = self.detector.detectAndCompute(
                    self.new_frame, None)
            self.kps = self.px_cur
            self.px_cur = np.array(
                    [x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert img.ndim==2, "Frame: given image isn't grayscale"
        assert img.shape[0]==self.cam.height and img.shape[1]==self.cam.width, \
                "Frame: geometry of given image doesn't match the camera model"
        self.last_frame = self.new_frame
        self.new_frame = img
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()

    def drive(self, max_frame, show=False):
        for img_id in range(max_frame+1):
            img_path = os.path.join(
                    self.dataset_path, 'gray', '00', 'image_0',
                    str(img_id).zfill(6)+'.png')
            img = cv2.imread(img_path, 0)
            self.update(img, img_id)
            if show:
                self.showDrive()
            sys.stderr.write(f"\rProcessing frame {img_id}")
            sys.stderr.flush()
        sys.stderr.write("\n")

    def showDrive(self):
        cv2.imshow('Road facing camera', self.new_frame)
        #
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            x, y, z = self.cur_t[0], self.cur_t[1], self.cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x)+290, int(z)+90
        true_x, true_y = int(self.trueX)+290, int(self.trueZ)+90
        cv2.circle(self.traj, (draw_x,draw_y), 1, (0, 0, 255), 1)
        cv2.circle(self.traj, (true_x,true_y), 1, (0, 255, 0), 1)
        cv2.rectangle(self.traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(self.traj, text,
                    (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Trajectory', self.traj)
        cv2.waitKey(1)

    def showImage(self, win_name, img):
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win_name, 0, 0)
        cv2.resizeWindow(win_name, *reversed(self.new_frame.shape[:2]))
        cv2.imshow(win_name, img)

    def visualiseFeature(self, feature_id, show=True):
        assert self.frame_stage == STAGE_DEFAULT_FRAME, \
                "Info on feature matches only available after second frame"
        prev_kp, cur_kp, matches = self.vt.featureMatching(
                self.last_frame, self.new_frame)
        visual_matches = cv2.drawMatchesKnn(
                self.last_frame, prev_kp, self.new_frame, cur_kp,
                matches[feature_id:feature_id+1], None, flags=2)
        if show:
            self.showImage('Feature Visual', visual_matches)
            cv2.waitKey(0)
        return visual_matches

    def visualiseFeatureInfo(self, feature_id, show=True):
        assert self.frame_stage == STAGE_DEFAULT_FRAME, \
                "Info on feature matches only available after second frame"
        prev_coord, cur_coords, cur_distances = self.getFeatureInfo(feature_id)
        hl_colour = (0, 255, 0)
        prev_image = cv2.cvtColor(self.last_frame, cv2.COLOR_GRAY2BGR)
        cur_image = cv2.cvtColor(self.new_frame, cv2.COLOR_GRAY2BGR)
        if len(cur_coords) > 0:
            prev_image = cv2.circle(
                    prev_image, prev_coord, 10, hl_colour, 2)
            for cur_coord in cur_coords:
                cur_image = cv2.circle(
                        cur_image, cur_coord, 10, hl_colour, 2)
        opencv_visual = self.visualiseFeature(feature_id, False)
        custom_visual = cv2.hconcat([prev_image, cur_image])
        visual_matches = cv2.vconcat([opencv_visual, custom_visual])
        if show:
            self.showImage('Feature Visual', visual_matches)
            cv2.waitKey(0)
        return visual_matches

    def getFeatureInfo(self, feature_id):
        assert self.frame_stage == STAGE_DEFAULT_FRAME, \
                "Info on feature matches only available after second frame"
        prev_coord, cur_coords, cur_distances = self.vt.featureInfo(
                self.last_frame, self.new_frame, feature_id)
        return prev_coord, cur_coords, cur_distances

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    import run_odometry
    run_odometry.main(sys.argv[1:])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vim:set et sw=4 ts=4:
