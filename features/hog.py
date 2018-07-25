from skimage import feature
import numpy as np
from PIL import Image


class HistgoramOrientedGradient(object):

    def __init__(self, orientation=8):
        super(HistgoramOrientedGradient, self).__init__()
        self.orientation = orientation

    def extract(self, img):
        """
        compute the histograms of Oriented Gradients 
        """
        hog = feature.hog(img, orientations=self.orientation,
                          block_norm='L1',
                          cells_per_block=(3, 3),
                          transform_sqrt=True)
        return hog

    def get_feature(self, image_path):
        img = Image.open(image_path).convert('L').resize((64, 64))
        return self.extract(img)


if __name__ == "__main__":

    model = HistgoramOrientedGradient()
    print(model.get_feature("../week2/cover.jpg"))
    print(model.get_feature('../week2/pic.jpg').shape)
    print(model.get_feature('../week2/zebra.png').shape)