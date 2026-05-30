import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os.path
from scipy.signal import find_peaks, savgol_filter
from .utils import resize_to_fit, map_bboxes_to_original, compute_iou, is_inside, retrieve_frames
from .video_reader import VideoReader
import time
from codecarbon import track_emissions
from tqdm import tqdm
import tracemalloc

class KeyFrameExtractor():
    """Extract key frames from a video.

    This class implements the functions needed to perform Key Frame Extraction (Selection)
    from a video:
    - back_project
    - good_features_to_track
    - lk_optical_flow
    - motion metrics

    The implementation of the back projection is adapted from the OpenCV tutorial code
    (https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo2.py),
    licensed under the Apache License 2.0.  See
    third_party/opencv/LICENSE or https://www.apache.org/licenses/LICENSE-2.0 for license details.

    """

    def __init__(self, model_data: str | None = None,
                 img_scale: float = 0.1,
                 bb_mean_width: int = 520,
                 bb_mean_height: int = 841,
                 bb_std_width: int = 71,
                 bb_std_height: int = 140):
        """Constructor. Loads the model data if the path is provided

        Arg:
            model_data: Path to a model histogram data file (.npz)
        """
        if model_data is not None:
            self.load_model_histogram_data(model_data)
        else:
            self.model_hist = None
            self.channels = None
            self.ranges = None

        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners = 100,
                                   qualityLevel = 0.3,
                                   minDistance = 7,
                                   blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize  = (15, 15),
                              maxLevel = 2,
                              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        self.scale = img_scale
        
        # Bounding boxes reference statistics.
        # Default values are computed on the EV2 dataset.
        self.bb_mean_width, self.bb_mean_height = bb_mean_width*self.scale, bb_mean_height*self.scale
        self.bb_std_width, self.bb_std_height = bb_std_width*self.scale, bb_std_height*self.scale
        # Bounding box outlier detection threshold, expressed in standard deviations
        self.bb_z_threshold = 2.0

        self.upper_duct = ((470, 40), (520, 200))
        self.lower_duct = ((115, 90), (145, 200))
        self.upper_duct_origin = (self.upper_duct[0][0] + (self.upper_duct[1][0] - self.upper_duct[0][0])//2,
                                  self.upper_duct[0][1]) # (495, 40)
        self.lower_duct_origin = (self.lower_duct[0][0] + (self.lower_duct[1][0] - self.lower_duct[0][0])//2,
                                  self.lower_duct[0][1]) # (70, 90)
        self.upper_duct_length = self.upper_duct[1][1] - self.upper_duct[0][1] # 160

        self.lateral_threshold = 50

    def compute_model_histogram(self, model_image:str, loDiff:int=128, upDiff:int=128):
        """Load the model image and set the parameters

        This function lets the user interactively set the parameters passed to cv.calcHist
        and cv.calcBackProject.

        """
        self.loDiff = loDiff
        self.upDiff = upDiff
        
        self.model_image = cv.imread(cv.samples.findFile(model_image))
        if self.model_image is None:
            print('Could not open or find the image:', model_image)
            exit(0)
        self.model_image, _ = resize_to_fit(self.model_image, 800, 600)
            
        hsv = cv.cvtColor(self.model_image, cv.COLOR_BGR2HSV)

        # Show the image
        window_image = 'Source image'
        cv.namedWindow(window_image)
        cv.imshow(window_image, self.model_image)
        
        # Set Trackbars for floodfill thresholds
        cv.createTrackbar('Low thresh', window_image, self.loDiff, 255, self.callback_loDiff)
        cv.createTrackbar('High thresh', window_image, self.upDiff, 255, self.callback_upDiff)
        # Set a Mouse Callback
        cv.setMouseCallback(window_image, self.pick_point)
        
        cv.waitKey()
        
    def pick_point(self, event, x, y, flags, param):
        """Compute the model histogram and back project the model image

        The histogram will be used as the model histogram when calling self.back_project.
        Choosing the right representative image (self.model_image) is fundamental and
        depends on the images that will be later fed to back_project().

        It is an interactive function. The user must click on the image to choose which
        pixels will contribute to the model histogram. The algorithm uses cv.floodFill to
        compute the connected component. When clicking on the image, the user effectively
        selects the seed passed to cv.floodFill. loDiff and upDiff are the homonymous
        cv.floodFill parameters.

        """
        if event != cv.EVENT_LBUTTONDOWN:
            return

        # Fill and get the mask
        seed = (x, y)
        newMaskVal = 255
        newVal = (120, 120, 120)
        connectivity = 8
        flags = connectivity + (newMaskVal << 8 ) + cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY

        mask2 = np.zeros((self.model_image.shape[0] + 2, self.model_image.shape[1] + 2), dtype=np.uint8)
        cv.floodFill(self.model_image, mask2, seed, newVal, (self.loDiff, self.loDiff, self.loDiff), (self.upDiff, self.upDiff, self.upDiff), flags)
        mask = mask2[1:-1,1:-1]

        cv.imshow('Mask', mask)
        self.calc_hist(mask)
        self.back_project(self.model_image)

    def calc_hist(self, mask):
        """Compute the model histogram.

        Given the mask obtained from cv.floodFill, compute the histogram of the masked
        pixels of self.model_image.

        """
        h_bins = 30
        s_bins = 32
        histSize = [h_bins, s_bins]
        h_range = [0, 180]
        s_range = [0, 256]
        self.ranges = h_range + s_range # Concat list
        self.channels = [0, 1]

        hsv = cv.cvtColor(self.model_image, cv.COLOR_BGR2HSV)
        
        # Get the Histogram and normalize it
        self.model_hist = cv.calcHist([hsv], self.channels, mask, histSize, self.ranges, accumulate=False)
        cv.normalize(self.model_hist, self.model_hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    def back_project(self, image: str | np.ndarray, imshow: bool = False):
        """Compute back projection on an image.

        Raises ValueError if self.model_hist, self.channels, and self.ranges are None.

        """
        if self.model_hist is None or self.channels is None or self.ranges is None:
            raise ValueError("model_hist, channels, and ranges must be set before calling KeyFrameExtractor.back_project. (Hint: call KeyFrameExtractor.compute_model_histogram)")
        
        if isinstance(image, str):
            image = cv.imread(cv.samples.findFile(image))
            image, _ = resize_to_fit(image, 800, 600)
        
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # Get back projection
        back_proj = cv.calcBackProject([hsv], self.channels, self.model_hist, self.ranges, scale=1)

        if imshow:
            # Draw the original image
            cv.imshow('Test image', image)

            # Draw the backproj
            cv.imshow('BackProj', back_proj)

            cv.waitKey()

        return back_proj

    def callback_loDiff(self, val):
        self.loDiff = val

    def callback_upDiff(self, val):
        self.upDiff = val

    def dump_model_histogram_data(self, path:str):
        """Dumpt the model histogram to a file for later use.

        This is a wrapper function for numpy.save. The dumped data are:
        1. the model histogram;
        2. the channels (e.g. hue and saturation);
        3. the ranges of the histogram bin boundaries in each dimension.
        """
        if self.model_hist is None or self.channels is None or self.ranges is None:
            raise RuntimeError("The model histogram data is not available: compute a model histogram first using KeyFrameExtractor.compute_model_histogram.")

        np.savez(path, model_hist=self.model_hist, channels=self.channels, ranges=self.ranges)

    def load_model_histogram_data(self, path:str):
        """Load the model histogram data to back project images.

        This is a wrapper function for numpy.load. The loaded data are:
        1. the model histogram;
        2. the channels (e.g. hue and saturation);
        3. the ranges of the histogram bin boundaries in each dimension.
        """
        npzfile = np.load(path)
        self.model_hist = npzfile['model_hist']
        self.channels = [int(x) for x in npzfile['channels']]
        self.ranges = [int(x) for x in npzfile['ranges']]

    def good_features_to_track_debug(self, frame: np.ndarray):
        back_proj = self.back_project(old_frame)
        p0 = cv.goodFeaturesToTrack(back_proj, mask = None, **self.feature_params)
        p = np.squeeze(p0).astype(int)
        for i, p_i in enumerate(p):
            old_frame = cv.circle(old_frame, p_i, 5, color[i].tolist(), -1)
        cv.imshow("Old Frame", old_frame)
        cv.waitKey()

    def findGoodFeaturesToTrack(self, back_proj, verbose: bool = False):
        """Find good features to track.

        This is a wrapper function for cv.goodFeaturesToTrack with the
        addition of a mechanism to adjust the quality requirements.

        Keyword arguments: back_proj -- the Back Projection of the
        frame verbose -- whether to print the update of the quality parameters
        """
        gftt_params = self.feature_params.copy()
        while(1):
            p0 = cv.goodFeaturesToTrack(back_proj, mask = None, **gftt_params)
            if p0 is not None:
                break
            else:
                if gftt_params["qualityLevel"] <= 0.1:
                    gftt_params["qualityLevel"] -= 0.01
                    if gftt_params["minDistance"] > 1:
                        gftt_params["minDistance"] -= 1
                else:
                    gftt_params["qualityLevel"] -= 0.1
                if gftt_params["qualityLevel"] < 0:
                    raise RuntimeError("qualityLevel parameter (goodFeaturesToTrack) is negative")
                if verbose:
                    print("Good Features To Track parameters updated:", gftt_params)
        return p0
        
    # @track_emissions
    def extract_frames(self, video:str,
                       num_key_frames: int | None = None,
                       show_optical_flow: bool = False,
                       show_bounding_box: bool = False,
                       show_metric: bool = False,
                       extract_orientation: bool = False,
                       verbose: bool = False,
                       show_bb_gt: bool = False,
                       squared: bool = False,
                       return_all_frames: bool = False,
                       return_bounding_boxes: bool = False,
                       return_iou: bool = False,
                       return_motion_metric: bool = False,
                       bb_gt: dict[int, tuple[tuple[int, int], ...]] = None) -> list[int]:
        """Extract key frames from a video.

        It extracts the key frames following the pipeline:
        1) back_project
        2) background_subtraction
        3) good_features_to_track
        4) lk_optical_flow
        5) compute_metric
        This approach is tailored to the EV2 dataset.
        It implements the solution proposed by Wayne Wolf in "Key Frame Selection by Motion Analysis".

        Keyword arguments:
        video -- the string of the video path
        num_key_frames -- preferred number of key frames to extract
        show_optical_flow -- show the optical flow on the playback of the video
        show_bounding_box -- show the bounding box on the playback of the video
        show_metric -- show the metrics at the end of the playback of the video
        verbose -- print debug messages
        show_bb_gt -- show the bounding box ground truth on the playback of the video
        return_all_frames -- return all extracted frames
        return_iou -- return the iou values for each frame
        bb_gt -- the ground truth of the bounding box
        """

        if verbose:
            print(f"Computing optical flow on video {video}")
        
        if not os.path.isfile(video):
            raise ValueError("Provide a valid video file path")

        cap = cv.VideoCapture(video)
        back_sub = cv.createBackgroundSubtractorMOG2()
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

        # motion metric
        motion_features = np.array([])

        # Read the entire video into memory
        video_reader = VideoReader(cap, scale=self.scale, return_original=False)
        cap.release()
        frames_iter = iter(video_reader)

        # Read first frame
        color = np.random.randint(0, 255, (100, 3))
        old_frame = next(frames_iter)
        old_back_proj = self.back_project(old_frame)
        p0 = cv.goodFeaturesToTrack(old_back_proj, mask = None, **self.feature_params)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        bounding_boxes = []
        cropping_data = []
        detected_frames = []

        if return_iou and bb_gt is not None:
            iou_data = {
                "iou": [],
                "bb_with_no_gt": 0
            }

        detected_in_last_frame = False
        frame_num = 1
        origin = None
        trajectory = []
        while(1):
            try:
                frame = next(frames_iter)
            except StopIteration:
                if verbose:
                    print('No more frames: reached end of video.')
                break

            back_proj = self.back_project(frame)
            fg_mask = back_sub.apply(frame)

            back_proj = cv.bitwise_and(back_proj, fg_mask)
            _, thresh = cv.threshold(back_proj, 30, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(thresh, kernel, iterations=5)
            contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            detected_in_this_frame = False
            boxes = []
            winning_bounding_box = None

            cnt_areas = {}
            # find bounding boxes, validate them, and extract the most promising
            for cnt in contours:
                area = cv.contourArea(cnt)
                x, y, w, h = cv.boundingRect(cnt)
                boxes.append((x, y, w, h))
                cnt_areas[(x, y, w, h)] = cnt
            valid_boxes = self.filter_valid_boxes(boxes)
            best_fit_box = self.get_best_fit_box(valid_boxes)
            # cv.rectangle(frame, self.upper_duct[0], self.upper_duct[1], (0, 255, 0), 2)
            # cv.rectangle(frame, self.lower_duct[0], self.lower_duct[1], (0, 255, 0), 2)
            if best_fit_box is not None:
                winning_cnt_area = cnt_areas[best_fit_box]
                M = cv.moments(winning_cnt_area)
                if M["m00"] != 0:
                    x_centroid = int(M["m10"] / M["m00"])
                    y_centroid = int(M["m01"] / M["m00"])
                else:
                    x_centroid, y_centroid = 0, 0

                centroid = (x_centroid, y_centroid)
                if origin is None:
                    # origin = np.mean(p0[st==1], axis=0)
                    origin = centroid
                cv.circle(frame, origin, radius=5, color=(0, 255, 255), thickness=-1)
                cv.circle(frame, centroid, radius=5, color=(0, 0, 255), thickness=-1)
                if np.abs(centroid[0] - origin[0]) < self.lateral_threshold:
                    centroid = (centroid[0] - origin[0], centroid[1] - origin[1])
                    trajectory.append(centroid)

            if best_fit_box is not None:
                detected_in_this_frame = True
                if not detected_in_last_frame:
                    detected_in_last_frame = True
                    if verbose:
                        print("Compute new Features To Track")
                    p0 = self.findGoodFeaturesToTrack(back_proj, verbose=verbose)
                winning_bounding_box = (best_fit_box[0],
                                        best_fit_box[1],
                                        best_fit_box[0] + best_fit_box[2],
                                        best_fit_box[1] + best_fit_box[3])
            if not detected_in_this_frame:
                """
                Keep track of frames without metric. Needed in
                order to have the absolute frame index when extracting
                key frames.  Frames with metric value -1 will be
                excluded during self.find_key_frames_robust.
                """
                detected_in_last_frame = False
                bounding_boxes.append(None)
                if not extract_orientation:
                    motion_features = np.append(motion_features, -1)
                # if not return_all_frames:
                # If not return_all_frames, then it only returns the key frames.
                # This None append will serve as a placeholder for the indexing of the
                # key frames.
                cropping_data.append(None)
            else:
                detected_frames.append(frame_num)
                bounding_boxes.append(winning_bounding_box)
                if squared:
                    start_h, start_w, dim = self.get_squared_bb(winning_bounding_box)
                    # original_frame = original_frame[start_h:start_h + dim, start_w:start_w + dim]
                    cropping = [start_h, start_h + dim, start_w, start_w + dim]
                    # if 0 not in original_frame.shape: cropped_frames.append(original_frame)
                    cropping_data.append(cropping)
                else:
                    x1, y1, x2, y2 = map_bboxes_to_original(winning_bounding_box, self.scale)
                    # original_frame = original_frame[y1:y2,x1:x2]
                    cropping = [y1, y2, x1, x2]
                    # if 0 not in original_frame.shape: cropped_frames.append(original_frame)
                    cropping_data.append(cropping)
                    
                p1, st, err = cv.calcOpticalFlowPyrLK(old_back_proj, back_proj, p0, None, **self.lk_params)
                
                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    # if len(good_new) >= 5:
                    flow_vectors = good_new - good_old
                    flow_metric = np.sum(np.linalg.norm(flow_vectors, axis=1))

                    if extract_orientation:
                        mean, eig = cv.PCACompute(good_new.astype(np.float32), mean=np.array([]))
                        theta = np.arctan2(eig[0, 1], eig[0, 0])  # radians
                        principal_vector = eig[0]  # first eigenvector = main orientation
                        scale = 50
                        end_point = (int(x_centroid + principal_vector[0]*scale), int(y_centroid + principal_vector[1]*scale))
                        # cv.arrowedLine(frame, (x_centroid, y_centroid), end_point, (255, 0, 0), 2, tipLength=0.2)
                        
                        sin_t = np.sin(theta)
                        cos_t = np.cos(theta)

                        if motion_features.shape[0] == 0:
                            motion_features = np.array([flow_metric, sin_t, cos_t])
                        else:
                            motion_features = np.vstack((motion_features,
                                                         np.array([flow_metric, sin_t, cos_t])))

                    else:
                        motion_features = np.append(motion_features, flow_metric)
                else:
                    motion_features = np.append(motion_features, -1)
                

                if np.any(st == 0):
                    # Some feature was lost, compute new features to track
                    if verbose:
                        print(f"A feature was lost: Compute New Features To Track")
                    if verbose:
                        print("back_proj", back_proj)
                    p0 = self.findGoodFeaturesToTrack(back_proj, verbose=verbose)
                    if verbose:
                        print("p0", p0)
                else:
                    # No feature lost, track old features
                    p0 = good_new.reshape(-1, 1, 2)

            if (return_iou and
                bb_gt is not None):
                if (frame_num in bb_gt and
                    winning_bounding_box is not None):
                    current_bb_gt = map_bboxes_to_original(bb_gt[frame_num], 1/self.scale)
                    flattened_bb_gt = (*current_bb_gt[0], *current_bb_gt[1])
                    iou = compute_iou(winning_bounding_box, flattened_bb_gt)
                    iou_data["iou"].append(iou)
                elif (frame_num not in bb_gt and
                      winning_bounding_box is not None):
                    iou_data["bb_with_no_gt"] += 1
                    
            if show_optical_flow:
                bp_frame = cv.cvtColor(back_proj, cv.COLOR_GRAY2BGR)
                cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
                cv.drawContours(bp_frame, contours, -1, (0, 255, 0), 2)
                cv.drawContours(dilated, contours, -1, (0, 255, 0), 2)
                if detected_in_this_frame:                    
                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                        bp_frame = cv.circle(bp_frame, (int(a), int(b)), 1, color[i].tolist(), -1)
                        frame = cv.circle(frame, (int(a), int(b)), 1, color[i].tolist(), -1)
                    bp_frame = cv.add(bp_frame, mask)
                    frame = cv.add(frame, mask)

                    if show_bounding_box:
                        # Draw rectangle on the image
                        cv.rectangle(bp_frame, winning_bounding_box[:2], winning_bounding_box[2:], (0, 255, 0), 2)
                        cv.rectangle(frame, winning_bounding_box[:2], winning_bounding_box[2:], (0, 255, 0), 2)

                        # Optional: label the box
                        cv.putText(bp_frame, f"Bee", (winning_bounding_box[0], winning_bounding_box[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                                   0.6, (0, 255, 0), 2)
                        cv.putText(frame, f"Bee", (winning_bounding_box[0], winning_bounding_box[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                                   0.6, (0, 255, 0), 2)

                        if show_bb_gt:
                            if frame_num in bb_gt:
                                curr_bb_gt = bb_gt[frame_num]
                                curr_bb_gt = map_bboxes_to_original(curr_bb_gt, 1/self.scale)
                                
                                # Draw ground truth bounding box
                                cv.rectangle(bp_frame, curr_bb_gt[0], curr_bb_gt[1], (0, 255, 255), 2)
                                cv.rectangle(frame, curr_bb_gt[0], curr_bb_gt[1], (0, 255, 255), 2)

                                # Optional: label the box
                                cv.putText(bp_frame, f"GT",
                                           (curr_bb_gt[0][0], curr_bb_gt[0][1] - 10),
                                           cv.FONT_HERSHEY_SIMPLEX,
                                           0.6, (0, 255, 255), 2)
                                cv.putText(frame, f"GT",
                                           (curr_bb_gt[0][0], curr_bb_gt[0][1] - 10),
                                           cv.FONT_HERSHEY_SIMPLEX,
                                           0.6, (0, 255, 255), 2)
                            else:
                                print(f"No ground truth bounding box available for frame {frame_num}")

                cv.imshow('Input Video', frame)
                cv.imshow('Optical Flow', bp_frame)
            """
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            """

            old_back_proj = back_proj.copy()
            frame_num += 1

        # cv.destroyAllWindows()

        if return_all_frames:
            cap = cv.VideoCapture(video)
            cropped_frames = retrieve_frames(cap, detected_frames, cropping_data)
            cap.release()
            return cropped_frames

        if return_motion_metric:
            trajectory = np.array(trajectory)
            return trajectory
        
        mask = motion_features != -1
        # No motion feature extracted
        if len(motion_features[mask]) == 0:
            raise RuntimeError(f"Key frame extraction on {video}: no bee movement detected")
            
        key_frames = self.find_key_frames_robust(self.smooth_motion_curve(motion_features[mask]), N_percent=0)
        
        # Back to the original indexing
        original_indices = np.where(mask)[0]
        key_frames = original_indices[key_frames].tolist()
        if show_metric:
            ### DEBUGGING REASON ###
            peaks, _ = find_peaks(motion_features[mask])
            peaks = original_indices[peaks].tolist()
            self.plot_metric(self.smooth_motion_curve(motion_features), peaks)
            ########################
            self.plot_metric(self.smooth_motion_curve(motion_features), key_frames)

        if verbose:
            print(f'kf: {key_frames}')
            print(f'bb: {[bounding_boxes[i] for i in key_frames]}')
            
        bounding_boxes_original_coords = map_bboxes_to_original([bounding_boxes[i] for i in key_frames], self.scale)
                    
        if num_key_frames is not None:
            key_frames, bounding_boxes_original_coords = self.get_best_fit_key_frames(key_frames, bounding_boxes_original_coords, num_key_frames)
            if len(key_frames) < num_key_frames:
                remaining_num_key_frames = num_key_frames - len(key_frames)
                mask_remaining = np.copy(mask)
                mask_remaining[key_frames] = False
                original_remaining_indices = np.where(mask_remaining)[0].astype(int).tolist()
                bounding_boxes_original_remaining_coords = map_bboxes_to_original([bounding_boxes[i] for i in original_remaining_indices], self.scale)
                remaining_key_frames, remaining_bb = self.get_best_fit_key_frames(original_remaining_indices, bounding_boxes_original_remaining_coords, remaining_num_key_frames)

                key_frames = key_frames + [i + 1 for i in remaining_key_frames]
                bounding_boxes_original_coords = bounding_boxes_original_coords + remaining_bb
        
        if return_iou:
            return key_frames, bounding_boxes_original_coords, motion_features, iou_data

        if return_bounding_boxes:
            return key_frames, bounding_boxes_original_coords, motion_features

        cap = cv.VideoCapture(video)
        cropped_frames = retrieve_frames(cap, key_frames, cropping_data)
        cap.release()
        # cropped_frames = [cropped_frames[key_frame] for key_frame in key_frames if cropped_frames[key_frame] is not None]
        return cropped_frames

    def get_squared_bb(self, bounding_box):
        """Return the start heigth, the start width, and the width of
        the box in the original size.
        
        """
        bb_original = map_bboxes_to_original(bounding_box, self.scale)
        frame_h = bb_original[3] - bb_original[1]
        frame_w = bb_original[2] - bb_original[0]
        max_dim = max(frame_h, frame_w)
        median_h = ((bb_original[3] - bb_original[1]) // 2) + bb_original[1]
        median_w = ((bb_original[2] - bb_original[0]) // 2) + bb_original[0]
        start_h = median_h - (max_dim // 2)
        start_w = median_w - (max_dim // 2)

        return start_h, start_w, max_dim

    
    def plot_metric(self,
                    M: np.ndarray,
                    key_frames: list[int] | None = None,
                    gt: dict | None = None):
        """ Plot speed and key frames

        Args:
            M: contains the metric value
            key_frames: list of key frames
            gt: dictionary containing the gt metric and key frames
        """
        plt.figure(figsize=(10, 4))
        plt.plot(M, label="Motion Metric")
        title = "Motion Metric Analysis"

        if key_frames is not None:
            # Overlay key frames
            plt.scatter(key_frames, [M[k] for k in key_frames], label="Key Frames", zorder=5)
            if gt is not None:
                gt_key_frames = np.array(gt['minima_indices'])
                # Overlay ground truth key frames
                plt.scatter(gt_key_frames, [M[k] for k in gt_key_frames], label="GT Key Frames", zorder=5)

                gt_speed = np.array(gt['speed'])*1e2
                plt.plot(gt_speed,  label="GT Speed")
            title = "Motion Metric Analysis with Key Frames"
        
        plt.title(title)
        plt.xlabel("Frame")
        plt.ylabel("Metric")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def smooth_motion_curve(self, M, window_length=11, polyorder=3):
        """Smooth the motion curve using Savitzky-Golay filter."""
        if len(M) < window_length:
            window_length = len(M) if len(M) % 2 == 1 else len(M) - 1
        if window_length < polyorder + 2:
            # savgol_filter requires window_length > polyorder
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
            if window_length > len(M):
                # Can't apply filter, return original
                return M
        return savgol_filter(M, window_length, polyorder)

    def find_key_frames_robust(self, M: np.ndarray, N_percent: int = 30, smoothing: bool = True) -> list[int]:
        """Select key frames based on the N% criterion.

        This function implements the key frame selection process
        described by W. Wolf in "Key Frame Selection by Motion
        Analysis" (Department of Electrical Engineeering, Princeton
        University, 1996): [...] The second step identifies local minima of
        M(t). Our algorithm scans the M(t) vs. t curve starting at
        t=O. It identifies two local maxima ml and m2 such that the
        value at m2 varies by at least N% from the M(t) value for
        mz. The local minimum of M(t} between these local maxima is
        selected as a key frame. The algorithm them selects m2 as the
        current ml and identifies the next m2.

        Args:
            m: the flow metric M(t) extracted from the video
            N: the N%, as described by W. Wolf

        Returns:
            The list of the key frames selected
        """
        
        peaks, _ = find_peaks(M)
        
        key_frames = []
        N = N_percent / 100.0
        i = 0

        while i < len(peaks) - 1:
            m1, m2 = peaks[i], peaks[i + 1]
            delta = abs(M[m2] - M[m1]) / max(M[m1], 1e-6)

            if delta >= N:
                # Find local minimum in between m1 and m2
                segment = M[m1:m2 + 1]
                if len(segment) == 0:
                    i += 1
                    continue
                min_idx = int(np.argmin(segment) + m1)
                key_frames.append(min_idx)
                i += 1  # Update m1 to m2
            else:
                i += 1  # Not enough variation
        return key_frames
    
    def extract_key_frames(self, video:str):
        """
        Extract key frames from a video.

        It extracts the key frames following the pipeline:
        1) back_project
        2) good_features_to_track
        3) lk_optical_flow
        4) compute_metric
        This approach is tailored to the EV2 dataset.
        It implements the solution proposed by Wayne Wolf in "Key Frame Selection by Motion Analysis".
        """
        if video is not None:
            self.cam = cv.VideoCapture(video)
        
            _ret, self.frame0 = self.cam.read()
        
            cv.namedWindow('optical flow')
            cv.resizeWindow('frame', 640, 480)

    def extract_cropped_frames(self, video_path, frame_indices, bboxes):
        """
        Extracts and crops specific frames from a video using bounding boxes.

        Args:
        video_path (str): Path to the video file.
        frame_indices (List[int]): Frame numbers to extract.
        bboxes (List[Tuple[int, int, int, int]]): Bounding boxes (x, y, w, h) for each frame.

        Returns:
        List[np.ndarray]: List of cropped frame regions.
        """
        if len(frame_indices) != len(bboxes):
            raise ValueError(f"frame_indices [{len(frame_indices)}] and bboxes [{len(bboxes)}] must have the same length.")
    
        cap = cv.VideoCapture(video_path)
        cropped_frames = []

        for frame_idx, (x_min, y_min, x_max, y_max) in tqdm(zip(frame_indices, bboxes)):
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                print(f"Warning: Could not read frame {frame_idx}. Skipping.")
                continue

            # Ensure crop box is within frame bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_end, y_end = min(frame.shape[1], x_max), min(frame.shape[0], y_max)
            cropped = frame[y_min:y_end, x_min:x_end]

            cropped_frames.append(cropped)

        cap.release()
        return cropped_frames

    def is_valid_box(self, w, h):
        z_w = abs((w - self.bb_mean_width) / self.bb_std_width)
        z_h = abs((h - self.bb_mean_height) / self.bb_std_height)
        return z_w < self.bb_z_threshold and z_h < self.bb_z_threshold

    def filter_valid_boxes(self, boxes):
        return [box for box in boxes if self.is_valid_box(box[2], box[3])]
    
    def box_distance(self, w, h):
        return ((w - self.bb_mean_width) / self.bb_std_width) ** 2 + ((h - self.bb_mean_height) / self.bb_std_height) ** 2

    def get_best_fit_box(self, valid_boxes):
        if not valid_boxes:
            return None
        return min(valid_boxes, key=lambda box: self.box_distance(box[2]-box[0], box[3]-box[1]))

    @staticmethod
    def top_k_key_frames_by_bb_size(data: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        return sorted_data[:k]
    
    def get_best_fit_key_frames(self, key_frames, boxes, n_frames):
        kf_boxes = list(zip(key_frames, boxes))
        box_dist = []
        
        for kf, box in kf_boxes:
            box_dist.append((kf, self.box_distance(box[2]-box[0], box[3]-box[1])))

        top_k_key_frames = KeyFrameExtractor.top_k_key_frames_by_bb_size(box_dist, n_frames)

        match_key_frames = {key_frame for key_frame, _ in top_k_key_frames}

        best_fit_key_frames = []
        best_fit_bbs = []
        for key_frame, bb in kf_boxes:
            if key_frame in match_key_frames:
                best_fit_key_frames.append(key_frame)
                best_fit_bbs.append(bb)

        return best_fit_key_frames, best_fit_bbs

    def flood_fill(self, mask):
        # Copy the mask and flood fill from the edges
        im_floodfill = mask.copy()
        h, w = mask.shape[:2]
        mask_flood = np.zeros((h+2, w+2), np.uint8)
        
        cv.floodFill(im_floodfill, mask_flood, (0,0), 255)  # fill from background
        
        # Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)
        
        # Combine original mask + filled holes
        filled = mask | im_floodfill_inv
        
        return filled
