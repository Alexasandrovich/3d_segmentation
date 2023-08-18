import cv2
import os
from tqdm import tqdm
import shutil

# todo: find optimal step i

def read_tifs(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(f'dtype: {img.dtype}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}')
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('tif example', img)
    cv2.waitKey(0)

    # some normalizing before saving


if __name__ == '__main__':
    path_to_source_imgs = "data/folded002.rec_2655/"
    where_to_save_converted = "data/preprocessed_data_for_train/hard_images"
    to_be_processed = sorted(os.listdir(path_to_source_imgs))
    save_as_is = True

    for i in tqdm(range(100, len(to_be_processed))):
        if i % 50 == 0:
            if save_as_is:
                shutil.copyfile(os.path.join(path_to_source_imgs, to_be_processed[i]),
                                os.path.join(where_to_save_converted, "%06d_scan.tif" % i))
            else:
                read_tifs(os.path.join(path_to_source_imgs, to_be_processed[i]))
