import matplotlib.pyplot as plt
from cellpose import models, metrics, plot, core
from cellpose.io_custom import imread
import numpy as np
from cellpose.plot import dx_to_circ
import cv2


def evel_metrics(label, pred, dP, flows):
    _, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY) # ?
    pred = (pred / 255).astype(int)
    label = label.astype(int)
    print(f"average_precision = {metrics.average_precision(label, pred)[0].mean():.3f}")
    print(f"flow_error = {metrics.flow_error(pred, dP)[0]}")
    # print(f"aggregated_jaccard_index = {metrics.aggregated_jaccard_index(label, pred)}")
    scales = np.arange(0.025, 0.275, 0.025)
    # print(f"boundary_scores = {metrics.boundary_scores(label, pred, scales)}")
    print(f"mask_ious = {metrics.mask_ious(label, pred)[0].mean()}")


def do_inference():
    # input
    img = imread("../coco_annotation/data/hard_images/test/002500_scan.tif")
    # 0 - flows, 1 - segmentation mask, 2 - x-gradient, 3 - y-gradient
    label = imread("../coco_annotation/data/hard_images/test/002500_scan_flows.tif")

    # model params
    model_path = 'models/cellpose_residual_on_style_on_concatenation_off_hard_images_2023_07_26_00_45_49.513595'
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')
    model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=model_path)
    print('model info:', model.pretrained_model)


    print('start predicting...')
    masks, flows, styles, dP = model.eval(img, diameter=None, channels=[[0, 0]], net_avg=False,
                                          progress=True, flow_threshold=0.0)

    # print some metrics
    evel_metrics(label[1], masks, dP, flows)

    plot.show_segmentation(img, masks, flows[0], label, channels=[[0, 0]])
    plt.show(block=False)


if __name__ == '__main__':
    do_inference()
