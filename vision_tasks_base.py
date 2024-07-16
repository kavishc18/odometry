
import cv2
import sys

NUM_FEATURES = 1500
MATCHING_ALGOS = "dt nn nndr".split()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VisionTasksBase:
    def dt(self, des1, des2, threshold):
        raise NotImplementedError

    def nn(self, des1, des2, threshold=None):
        raise NotImplementedError

    def nndr(self, des1, des2, threshold):
        raise NotImplementedError

    def matching_info(self, prev_image, cur_image, feature_id):
        raise NotImplementedError

    def __init__(self, matching_algo=None, threshold=None):
        self.matching_algo = None
        for algo in MATCHING_ALGOS:
            if algo == matching_algo:
                self.matching_algo = getattr(self, algo)
                break
        self.threshold = threshold
        # The following line creates a SIFT detector
        self.detector = cv2.SIFT_create(nfeatures=NUM_FEATURES)

    lk_params = dict(
        winSize  = (21, 21),
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def featureTracking(self, prev_image, cur_image, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(
                prev_image, cur_image, px_ref, None, **self.lk_params)
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        return kp1, kp2

    def featureMatching(self, prev_image, cur_image):
        # The following lines return a list of keypoints (kp)
        # and a list of descriptors (des) for a given image
        prev_kp, prev_des = self.detector.detectAndCompute(prev_image,None)
        cur_kp, cur_des = self.detector.detectAndCompute(cur_image,None)
        matches = self.matching_algo(prev_des, cur_des, self.threshold)
        assert type(matches) is list, \
            "Matching algorithm must return a list of matches"
        return prev_kp, cur_kp, matches

    def featureInfo(self, prev_image, cur_image, feature_id):
        prev_kp, cur_kp, matches = self.featureMatching(prev_image, cur_image)
        if matches:
            prev_coord, cur_coords, cur_distances = self.matching_info(
                    prev_kp, cur_kp, matches[feature_id])
        else:
            prev_coord, cur_coords, cur_distances = (0,0), [], []
        assert type(prev_coord) is tuple, \
            "Matching info must return the coordinates of feature"
        assert type(cur_coords) is list, \
            "Matching info must return a list of coordinates for feature matches"
        assert type(cur_distances) is list, \
            "Matching info must return a list of distances for feature matches"
        return prev_coord, cur_coords, cur_distances

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    import run_odometry
    run_odometry.main(sys.argv[1:])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vim:set et sw=4 ts=4:
