import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot
from cellpose.io import imread
import numpy as np

from cellpose.plot import dx_to_circ


def do_inference():
    img = imread("test_data/000150_scan.tif")
    label = imread("test_data/000150_scan_flows.tif")

    model_path = 'models/cellpose_residual_on_style_on_concatenation_off_hard_images_2023_08_13_22_48_good_dataset'
    model = models.CellposeModel(gpu=False, pretrained_model=model_path)
    print('model info:', model.pretrained_model)
    print('start predicting...')

    masks, flows, styles, row_output = model.eval(img, diameter=None, channels=[[0, 0]], net_avg=False,
                                                  progress=True, flow_threshold=0.0)

    plot.show_segmentation(img, masks, flows[0], label, channels=[[0, 0]])
    plt.show()


if __name__ == '__main__':
    do_inference()
