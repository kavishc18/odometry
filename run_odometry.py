
import argparse
import cv2
import os.path
import sys

from vision_tasks_base import NUM_FEATURES, MATCHING_ALGOS
from vision_tasks import VisionTasks
from visual_odometry import PinholeCamera, VisualOdometry

MAX_FRAME = 500
DATASET_PATH = os.path.expanduser('~/MyKITTI')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RunOdometry:
    def __init__(self,
             matching_algo=None, threshold=None, dataset_path=DATASET_PATH):
        vt = VisionTasks(matching_algo, threshold)
        cam = PinholeCamera()
        self.vo = VisualOdometry(vt, cam, dataset_path)

    def view_trajectory(self, frame_id=MAX_FRAME):
        self.vo.drive(frame_id, True)

    def view_feature(self, frame_id, feature_id):
        self.vo.drive(frame_id)
        opencv_visual = self.vo.visualiseFeature(feature_id)
        cv2.imwrite('opencv_visual.png', opencv_visual)

    def view_info(self, frame_id, feature_id):
        self.vo.drive(frame_id)
        visual_matches = self.vo.visualiseFeatureInfo(feature_id)
        cv2.imwrite('custom_visual.png', visual_matches)

    def get_info(self, frame_id, feature_id):
        self.vo.drive(frame_id)
        return self.vo.getFeatureInfo(feature_id)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(*argv):
    def add_ro_arguments(sp, title, *args, **kwargs):
        p = sp.add_parser(title, *args, **kwargs)
        p.add_argument(
            '-a', '--algorithm', action='store', required=True, type=str,
            choices=MATCHING_ALGOS,
            help='name of the matching algorithm used to match features')
        p.add_argument(
            '-t', '--threshold', action='store', default=None, type=float,
            required=not 'nn' in sys.argv,
            help='decimal value of threshold for feature matching \
                    (option required except for nn algorithm)')
        p.add_argument(
            'frame_id', type=int,
            help=f"index of chosen frame (1 to {MAX_FRAME})")
        p.add_argument(
            'feature_id', type=int,
            help=f"index of chosen feature (0 to {NUM_FEATURES-1})")
        return p
    #
    p0 = argparse.ArgumentParser()
    p0.add_argument(
            '-d', '--dataset', action='store', default=DATASET_PATH,
            required=not os.path.exists(DATASET_PATH),
            help=f"path to KITTI dataset directory (option required \
                    unless dataset is located at {DATASET_PATH})")
    sp = p0.add_subparsers(
            dest='command', required=True,
            description='select which odometry command to run')
    #
    p1 = sp.add_parser(
            'view_trajectory',
            help='show the car camera view and calculated trajectory')
    p1.add_argument(
            'frame_id', nargs='?', default=MAX_FRAME, type=int,
            help=f"index of frame to stop visualisation (1 to {MAX_FRAME})")
    #
    add_ro_arguments(
            sp, 'view_feature',
            help='show the matches for a frame feature using OpenCV \
                    (and save this image as opencv_visual.png)')
    add_ro_arguments(
            sp, 'view_info',
            help='show the matches for a frame feature \
                    using both OpenCV and the info algorithm \
                    (and save this composite image as custom_visual.png)')
    add_ro_arguments(
            sp, 'get_info',
            help='use the info algorithm \
                    to get details of matches for a frame feature')
    #
    config = p0.parse_args(*argv)
    params = []
    for attr in "algorithm threshold dataset".split():
        params.append(getattr(config, attr, None))
    args = []
    for attr in "frame_id feature_id".split():
        if hasattr(config, attr):
            args.append(getattr(config, attr))
    print("vo params: {}".format( (*params,) ))
    print("debug run: {}{}".format(config.command, (*args,)))
    run = RunOdometry(*params)
    app = getattr(run, config.command)
    ret = app(*args)
    print("ret value:", ret)
    try:
      cnt = len(ret)
      print("ret count:", cnt)
    except TypeError:
      pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main(sys.argv[1:])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vim:set et sw=4 ts=4:
