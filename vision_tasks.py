

import cv2
import sys

from vision_tasks_base import VisionTasksBase

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VisionTasks(VisionTasksBase):
    def __init__(self, *params):
        """Initialise instance by passing arguments to super class"""
        super().__init__(*params)

    def dt(self, des1, des2, threshold):
        """Implements feature matching based on distance thresholding

        :param des1: descriptors for the previous image (query)
        :type des1:  list
        :param des2: descriptors for the current image (train)
        :type des2:  list
        :param threshold: threshold value
        :type threshold:  float

        :return: matches for descriptors
        :rtype:  list
        """
        bf = cv2.BFMatcher()
        # Finding the top 2 matches for each descriptor
        matches = bf.knnMatch(des1, des2, k=1500)
        good_matches = []

        for m in matches:
            good_matches.append([])
            for n in m:
                if n.distance <= threshold:  # Check if the closest match is within the threshold
                    good_matches[len(good_matches)-1] . append(n)

        return good_matches

    def nn(self, des1, des2, threshold=None):
        """Implements feature matching based on nearest neighbour

        :param des1: descriptors for the previous image (query)
        :type des1:  list
        :param des2: descriptors for the current image (train)
        :type des2:  list
        :param threshold: threshold value
        :type threshold:  float or None

        :return: matches for descriptors
        :rtype:  list
        """
        bf = cv2.BFMatcher()
        # Finding the top 2 matches for each descriptor
        matches = bf.knnMatch(des1, des2, k=1)
        good_matches = []

        for m in matches:
            good_matches.append([])
            for n in m:
                if threshold is None or n.distance <= threshold:  # Check if the closest match is within the threshold
                    good_matches[len(good_matches)-1] . append(n)

        return good_matches

    def nndr(self, des1, des2, threshold):
        """Implements feature matching based on nearest neighbour distance ratio

        :param des1: descriptors for the previous image (query)
        :type des1:  list
        :param des2: descriptors for the current image (train)
        :type des2:  list
        :param threshold: threshold value
        :type threshold:  float

        :return: matches for descriptors
        :rtype:  list
        """
        bf = cv2.BFMatcher()
        # Finding the top 2 matches for each descriptor
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []

        for m in matches:
            good_matches.append([])

            dist1 = m[0].distance
            dist2 = m[1].distance

            if dist2 == 0:
                continue
            distratio = dist1/dist2
            if distratio < threshold:
                good_matches[len(good_matches)-1] .append(m[0])
        return good_matches


    def matching_info(self, kp1, kp2, feature_matches):
        """Collects information about the matches of some feature

        :param kp1: keypoints for the previous image (query)
        :type kp1:  list
        :param kp2: keypoints for the current image (train)
        :type kp2:  list
        :param feature_matches: matches for the feature
        :type feature_matches:  list

        :return: coordinate of feature in previous image,
                 coordinates for feature matches in current image,
                 distances for feature matches in current image
        :rtype:  tuple, list, list
        """


        cur_coords = []
        distances = []


        if feature_matches is None:
                return (0, 0), cur_coords, distances
        
        prev_coords = None
        for match in feature_matches:
            ind1 = match.queryIdx
            # ind1 = match.queryIdx

            prev_keypoint = kp1[ind1]
            cur_keypoint = kp2[match.trainIdx]

            x_prev, y_prev = int(prev_keypoint.pt[0]), int(prev_keypoint.pt[1])
            x_cur, y_cur = int(cur_keypoint.pt[0]), int(cur_keypoint.pt[1])

            prev_coords = (x_prev, y_prev)
            cur_coords.append((x_cur, y_cur))
            distances.append(match.distance)

        if prev_coords is None:
            return (0, 0), cur_coords, distances

        return prev_coords, cur_coords, distances
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    import run_odometry
    run_odometry.main(sys.argv[1:])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vim:set et sw=4 ts=4:
