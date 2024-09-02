import json
from copy import deepcopy

import cv2
import numpy as np
import torch
from collections import namedtuple
from skimage.morphology import convex_hull_image

from .pose_utils import draw_poses, decode_json_as_poses
from .util import HWC3

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


def draw_rectangle_torch(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle on a torch tensor image
    :param image: tensor image
    :param bbox: [x1, y1, x2, y2]
    :param color: color of the rectangle
    :param thickness: thickness of the rectangle
    :return: image with rectangle
    """
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    x1, y1, x2, y2 = bbox
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image


def dilate_mask(mask, n):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(mask, kernel, iterations=n)
    return dilated_image


def get_visible_keypoints(keypoints_list, keep_only: list = None):
    """
    Get visible keypoints from keypoints_list that has the form [x1, y1, c1, x2, y2, c2, ...]
    :param keypoints_list: OpenPose JSON output
    :param keep_only: list of indices to keep, if None keep all
    :return: list of visible keypoints
    """
    if keypoints_list is None:
        return []
    if keep_only is not None:
        keep_only = set(keep_only)
    else:
        keep_only = set(range(len(keypoints_list) // 3))
    visible_points = []
    for i, (x, y, c) in enumerate(zip(keypoints_list[0::3], keypoints_list[1::3], keypoints_list[2::3])):
        if i in keep_only:
            if c == 1:
                visible_points.append((x, y))
    return visible_points


def filter_poses(pose_keypoints, n_poses):
    """
    Filter poses by size, include only the n_poses largest poses and also sort them left to right
    :param pose_keypoints: OpenPose JSON output
    :param n_poses: int, number of poses to keep
    :return: pose_keypoints with only n_poses largest poses, sorted
    """
    pose_keypoints = deepcopy(pose_keypoints)
    sizes = []
    for i, person in enumerate(pose_keypoints[0]["people"]):
        points = person["pose_keypoints_2d"]
        # x, y, confidence
        leftmost = pose_keypoints[0]['canvas_width']
        rightmost = 0
        topmost = pose_keypoints[0]['canvas_height']
        bottommost = 0
        visible_points = len([c for c in points[2::3] if c == 1])
        if visible_points < 3:
            continue
        for x, y, c in zip(points[0::3], points[1::3], points[2::3]):
            if c == 1:
                if x > rightmost:
                    rightmost = x
                if x < leftmost:
                    leftmost = x
                if y > bottommost:
                    bottommost = y
                if y < topmost:
                    topmost = y
        sizes.append((i, (leftmost - rightmost) * (topmost - bottommost), (leftmost + rightmost) // 2))
    sizes = sorted(sizes, reverse=True, key=lambda ss: ss[1])[:n_poses]
    sizes = sorted(sizes, key=lambda ss: ss[2])
    new_people = []
    for i, _, _ in sizes:
        new_people.append(pose_keypoints[0]["people"][i])
    pose_keypoints[0]["people"] = new_people
    return pose_keypoints


def filter_masks(masks, face_bbox: list[list[int]]):
    """
    Filter masks to only include the faces in face_bbox regions
    :param masks: list of mask images
    :param face_bbox: list of bboxes, [[x1, y1, x2, y2], ...] - upper left and bottom right corners of the face bounding box
    """
    if face_bbox is None:
        print("No face bbox")
        return masks, list(range(len(masks)))
    new_masks = []
    idxs = []
    for bbox in face_bbox:
        best_idx = -1
        best_coverage = 0
        for i, mask in enumerate(masks):
            bbox_img = np.zeros_like(mask, dtype=np.uint8)
            bbox_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            intersection = np.logical_and(mask, bbox_img)
            union = np.logical_or(mask, bbox_img)
            size = np.sum(intersection) / np.sum(union)
            if size > best_coverage:
                best_coverage = size
                best_idx = i
        if best_idx != -1:
            new_masks.append(masks[best_idx])
            idxs.append(best_idx)
    return new_masks, idxs


def mapping_compositum(first: list, second: list):
    new_mapping = [-1] * len(first)
    for i, m1 in enumerate(first):
        if m1 == -1:
            continue
        if m1 > len(second):
            continue
        new_mapping[i] = second[m1]
    return new_mapping


def mapping_bbox_masks(face_bbox: list[list[int]], masks):
    if face_bbox is None:
        print("No face bbox")
        return masks, list(range(len(masks)))
    mapping = [-1] * len(face_bbox)
    for n, bbox in enumerate(face_bbox):
        best_idx = -1
        best_coverage = 0
        for i, mask in enumerate(masks):
            bbox_img = np.zeros_like(mask, dtype=np.uint8)
            bbox_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            intersection = np.logical_and(mask, bbox_img)
            union = np.logical_or(mask, bbox_img)
            size = np.sum(intersection) / np.sum(union)
            if size > best_coverage:
                best_coverage = size
                best_idx = i
        if best_idx != -1:
            mapping[n] = best_idx
    return mapping


def complete_mapping(mapping: list, max_num: int):
    """
    :param mapping: list of correspondences [-1, 3,2,4,-1,1]
    :param max_num: maximum number in the map
    :return: mapping with reduced -1s - use avaliable spots if any - up to max_num
    """
    new_mapping = mapping.copy()
    remaining_vals = set(range(max_num + 1)) - set(mapping)
    for i in range(len(mapping)):
        if mapping[i] != -1:
            continue
        if len(remaining_vals) == 0:
            return mapping
        new_val = remaining_vals.pop()
        new_mapping[i] = new_val
    return new_mapping


class MaskFromPoints:
    def __init__(self):
        self.keep_only_points = {
            'full-body': None,
            'face+shoulders': [0, 1, 2, 5, 14, 15, 16, 17],
            'face+torso': [0, 1, 2, 5, 14, 15, 16, 17, 8, 11],
            'face': [0, 14, 15, 16, 17]
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "use_keypoints": (["full-body", "face+shoulders", "face+torso", "face"], {'default': 'face+shoulders'}),
                "mask_width": ("INT", {
                    "default": 1,
                    "min": 0,  # Minimum value
                    "max": 2000,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "mask_height": ("INT", {
                    "default": 1,
                    "min": 0,  # Minimum value
                    "max": 2000,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "n_poses": ("INT", {
                    "default": 1,
                    "min": 1,  # Minimum value
                    "max": 10,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "dilate_iterations": ("INT", {
                    "default": 1,
                    "min": 0,  # Minimum value
                    "max": 30,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
            "optional": {
                "mask_mapping": ("MAPPING",),
                "face_bbox": ("BBOXLIST",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK1", "MASK2", "MASK3", "MASK4", "MASK5")
    FUNCTION = "poses_to_masks"
    CATEGORY = "Personal Tools"

    def poses_to_masks(self, pose_keypoint, use_keypoints, mask_width, mask_height, n_poses, dilate_iterations,
                       mask_mapping=None, face_bbox=None):
        # mask sizes should be the same as the bbox source image
        height = pose_keypoint[0]['canvas_height']
        width = pose_keypoint[0]['canvas_width']
        poses = pose_keypoint
        if not mask_mapping or not face_bbox:
            poses = filter_poses(pose_keypoint, n_poses)
        # 1. Get all masks for skeletons
        all_masks = []
        for person in poses[0]["people"]:
            body_points = person["pose_keypoints_2d"]
            face_points = person["face_keypoints_2d"]
            visible_body = get_visible_keypoints(body_points, keep_only=self.keep_only_points[use_keypoints])
            visible_face = get_visible_keypoints(face_points)
            visible_points = visible_body + visible_face
            img = np.zeros((height, width), dtype=np.uint8)
            # clip points to image size, if they are outside (may happen with dwpose)
            for x, y in visible_points:
                if y >= height:
                    y = height - 1
                if y < 0:
                    y = 0
                if x >= width:
                    x = width - 1
                if x < 0:
                    x = 0
                img[int(y), int(x)] = 1
            mask = convex_hull_image(img).astype(np.uint8)
            if dilate_iterations > 0:
                mask = dilate_mask(mask, dilate_iterations)
            mask = cv2.resize(mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
            all_masks.append(mask)

        full_mapping = list(range(len(all_masks)))
        if face_bbox:
            full_mapping = mapping_bbox_masks(face_bbox, all_masks)
            if mask_mapping:
                full_mapping = mapping_compositum(mask_mapping, full_mapping)
        full_mapping = complete_mapping(full_mapping, len(all_masks) - 1)

        all_masks_mapped = []
        for m in full_mapping:
            if m == -1:
                all_masks_mapped.append(np.zeros((mask_height, mask_width), dtype=np.float32))
            else:
                all_masks_mapped.append(all_masks[m])

        poses_decoded, _, _ = decode_json_as_poses(poses[0], normalize_coords=True)
        poses_decoded = [poses_decoded[k] for k in full_mapping if k != -1]
        pose_image = draw_poses(poses_decoded, height, width, draw_body=True, draw_face=True, draw_hand=False)
        pose_image = HWC3(pose_image)
        pose_image = torch.from_numpy(pose_image.astype(np.float32) / 255.0).unsqueeze(0)
        while len(all_masks_mapped) < 5:
            all_masks_mapped.append(np.zeros((height, width), dtype=np.float32))
        for i in range(5):
            # all_masks_mapped[i] = all_masks_mapped[i].astype(np.float32)
            all_masks_mapped[i] = torch.tensor(all_masks_mapped[i]).unsqueeze(0)
        return (pose_image, all_masks_mapped[0],
                all_masks_mapped[1], all_masks_mapped[2], all_masks_mapped[3], all_masks_mapped[4])


class FilterPoses:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "n_poses": ("INT", {
                    "default": 1,
                    "min": 1,  # Minimum value
                    "max": 10,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("image", "POSE_KEYPOINT")
    FUNCTION = "filter_poses"
    CATEGORY = "Personal Tools"

    def filter_poses(self, pose_keypoint, n_poses):
        height = pose_keypoint[0]['canvas_height']
        width = pose_keypoint[0]['canvas_width']
        poses = filter_poses(pose_keypoint, n_poses)
        poses_decoded, _, _ = decode_json_as_poses(poses[0], normalize_coords=True)
        pose_image = draw_poses(poses_decoded, height, width, draw_body=True, draw_face=True, draw_hand=False)
        pose_image = HWC3(pose_image)
        pose_image = torch.from_numpy(pose_image.astype(np.float32) / 255.0).unsqueeze(0)
        return pose_image, poses


class LoadPosesJSON:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint", "POSE_KEYPOINT")
    FUNCTION = "load_poses_json"
    CATEGORY = "Personal Tools"

    def validate_poses(self, pose_keypoint):
        if isinstance(pose_keypoint, list):
            pose_keypoint = pose_keypoint[0]
        if "people" not in pose_keypoint:
            raise ValueError("people not found")
        for person in pose_keypoint["people"]:
            if "pose_keypoints_2d" not in person:
                raise ValueError("pose_keypoints_2d not found")
            if len(person["pose_keypoints_2d"]) != 54:
                raise ValueError("pose_keypoints_2d should have 54 numbers")
        if "canvas_width" not in pose_keypoint:
            raise ValueError("canvas_width not found")
        if "canvas_height" not in pose_keypoint:
            raise ValueError("canvas_height not found")
        return True

    def load_poses_json(self, pose_keypoint):
        pose_dict = json.loads(pose_keypoint)
        self.validate_poses(pose_dict)
        if not isinstance(pose_dict, list):
            pose_dict = [pose_dict]
        return (pose_dict,)


class MaskListToMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("MASK1", "MASK2", "MASK3", "MASK4", "MASK5")
    FUNCTION = "mask_list_to_mask"
    CATEGORY = "Personal Tools"

    def mask_list_to_mask(self, mask):
        if len(mask) == 0:
            return (torch.zeros((1, 512, 512)),) * 5
        _, w, h = mask[0].shape
        missing = 5 - len(mask)
        if missing > 0:
            mask += [torch.zeros((1, w, h)) for _ in range(missing)]
        return tuple(mask)


class SegsListToSegs:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("SEGS", "SEGS", "SEGS", "SEGS", "SEGS")
    RETURN_NAMES = ("SEGS1", "SEGS2", "SEGS3", "SEGS4", "SEGS5")
    FUNCTION = "segs_list_to_segs"
    CATEGORY = "Personal Tools"

    def segs_list_to_segs(self, segs):
        if len(segs) == 0:
            empty_seg = ((1024, 1024), [])
            return (empty_seg,) * 5

        w, h = segs[0]
        seglist = segs[1]
        missing = 5 - len(seglist)
        new_segs = []
        for seg in seglist:
            if seg is None:
                new_segs.append(None)
            else:
                new_segs.append(((w, h), [seg]))
        while missing > 0:
            new_segs.append(((1024, 1024), []))
            missing -= 1
        return tuple(new_segs)


class BBOXSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bbox": ("BBOXLIST",),
                "idx": ("INT", {
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 9999,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
            "optional": {
                "mapping": ("MAPPING",),
            }
        }

    RETURN_TYPES = ("BBOXLIST",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "select_bbox"
    CATEGORY = "Personal Tools"

    def select_bbox(self, bbox, idx, mapping=None):
        print(f"N bboxes: {len(bbox)}")
        if mapping is None:
            mapping = list(range(len(bbox)))
        if len(bbox) == 0:
            print('No bbox')
            return ([],)
        if len(mapping) == 0:
            print("empty mapping")
            return (bbox,)
        if idx >= len(mapping):
            return []
        if mapping[idx] == -1:
            return ([],)
        if mapping[idx] >= len(bbox):
            print("mapping out of range")
            return ([],)
        res = ([bbox[mapping[idx]]],)
        print(f"BBOX selected: {res}")
        return res


class PreviewBBOX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bbox": ("BBOXLIST",),
                "width": ("INT", {
                    "default": 1024,
                    "min": 0,  # Minimum value
                    "max": 2000,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 0,  # Minimum value
                    "max": 2000,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "preview_bbox"
    OUTPUT_NODE = True
    CATEGORY = "Personal Tools"

    def preview_bbox(self, bbox, width, height, image=None):
        for b in bbox:
            if len(b) != 4:
                raise ValueError(f"BBOX should have 4 elements, {b}")
        if image is None:
            image = torch.zeros(1, height, width, 3)
        for b in bbox:
            image = draw_rectangle_torch(image, b)
        return (image,)
