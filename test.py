import json
import os

import numpy as np
from PIL import Image

from nodes import MaskFromPoints

script_dir = os.path.dirname(__file__)


def main():
    keypoints = json.load(open(script_dir + '/poses_example_02.json'))
    ff = MaskFromPoints()
    masks = ff.poses_to_masks(keypoints, 3, 10)
    m = masks[0][0].cpu().numpy() * 255
    m = m.astype(np.uint8)
    img = Image.fromarray(m)
    img.show()


if __name__ == '__main__':
    main()
