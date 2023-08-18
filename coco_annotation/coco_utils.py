import json
import urllib.request
import numpy as np
import cv2


def load_COCO_sample(annotation_path: str):
    # download the markup file
    with open(annotation_path, 'r') as f:
        data = json.load(f)

        # random image selection
        image_info = np.random.choice(data['images'])

        # image loading
        url = image_info['coco_url']
        img = urllib.request.urlopen(url)
        img = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Selecting objects in the image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_info['id']]

        # adding outlines of objects to the image
        for ann in annotations:
            seg = ann['segmentation'][0]
            poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.polylines(img, [poly], True, (0, 255, 0), 2)

        # save
        cv2.imwrite('result.jpg', img)


if __name__ == '__main__':
    load_COCO_sample('data/COCO_samples/instances_train2017.json')
