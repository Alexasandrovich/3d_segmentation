import cv2
import json
import numpy as np
from random import randint
import os
from tqdm import tqdm


# todo: add UnitTests

def backward_convertation_annotation(annotation_path: str, where_save: str, original_image_path: str, image_id: int):
    orig_image = cv2.imread(original_image_path)
    img = np.zeros((orig_image.shape[0], orig_image.shape[1], 3), dtype=np.uint8)
    with open(annotation_path, "r") as json_file:
        data = json.loads(json_file.read())
        # Selecting objects in the image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        # adding outlines of objects to the image
        for ann in annotations:
            seg = ann['segmentation'][0]
            poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
            prev_pt = None
            for cur_pt in poly:
                if prev_pt is not None:
                    cv2.line(img, prev_pt, cur_pt, (255, 255, 255), 1)
                prev_pt = cur_pt

    # save
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(where_save, img)


def convertation_to_coco(segm_path: str, image_id: int, bin_threshold_value: int = 60, approx_throshold: int = 1,
                         where_save_original: str = None):
    image = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

    # binarization
    ret, binary = cv2.threshold(image, bin_threshold_value, 255, cv2.THRESH_BINARY)
    if where_save_original is not None:
        cv2.imwrite(where_save_original % image_id, binary)

    # find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Coarse contours
    coarsed_contours = []
    for i in range(len(contours)):
        contour = np.squeeze(contours[i])
        contour = cv2.approxPolyDP(contour, approx_throshold, True)  # Duglas-Peker
        contour = contour.tolist()
        if len(contour) >= 3:
            polygon = []
            for point in contour:
                x, y = point[0]
                polygon.append(x)
                polygon.append(y)
            coarsed_contours.append(polygon)

    # MS COCO annotations
    annotation = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [{
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": coarsed_contours,
            "area": 0,
            "bbox": [],
            "iscrowd": 0
        }],
        "categories": [{
            "id": 1,
            "name": "segmentation",
            "supercategory": ""
        }]
    }

    return annotation



if __name__ == '__main__':
    # CT_OCR_2022
    # outside bound == 150 pixel value
    # inside bound == 200 pixel value
    # 3D body == 70

    path_to_seg_labels = "data/folded002.seg_2655/"
    where_to_save_converted = "data/preprocessed_data_for_train/hard_images"
    save_bin_original = "data/preprocessed_data_for_train/hard_images"
    to_be_processed = sorted(os.listdir(path_to_seg_labels))
    where_save_original = os.path.join(save_bin_original, "%06d_seg.png")
    need_test = True
    processed = []

    for i in tqdm(range(100, len(to_be_processed))):
        if i % 50 == 0:
            with open(os.path.join(where_to_save_converted, "%06d_seg_as_coco_lines.json" % i), "w") as annotation:
                json.dump(convertation_to_coco(segm_path=os.path.join(path_to_seg_labels, to_be_processed[i]),
                                               image_id=i,
                                               where_save_original=where_save_original),
                          annotation)
                processed.append(os.path.join(where_to_save_converted, "%06d_seg_as_coco_lines.json" % i))

            if need_test:
                backward_convertation_annotation(processed[-1],
                                                 os.path.join(where_to_save_converted, "%06d_seg_as_pix.png" % i),
                                                 os.path.join(path_to_seg_labels, to_be_processed[i]),
                                                 i)
