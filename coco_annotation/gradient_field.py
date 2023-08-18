import cv2
import os
from tqdm import tqdm


def convertation_annotations_to_flow_repr(input_path: str, where_save: str):
    # Load the input image and make it grayscale.
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # field calculation to the boundary
    dist_transform = cv2.distanceTransform(gray, cv2.DIST_L2, 3)

    # Normalization for easy visualization
    cv2.normalize(dist_transform, dist_transform, 0, 255.0, cv2.NORM_MINMAX)
    cv2.imwrite(where_save, dist_transform)


if __name__ == '__main__':
    # CT_OCR_2022
    # outside bound == 150 pixel value
    # inside bound == 200 pixel value
    # 3D body == 70

    path_to_seg_labels = "data/folded002.seg_2655/"
    where_to_save_converted = "data/preprocessed_data_for_train/hard_images/"
    to_be_processed = sorted(os.listdir(path_to_seg_labels))

    for i in tqdm(range(100, len(to_be_processed))):
        if i % 50 == 0:
            convertation_annotations_to_flow_repr(os.path.join(path_to_seg_labels, to_be_processed[i]),
                                                  os.path.join(where_to_save_converted, "%06d_scan_flows.png" % i))
