import numpy as np
import os
from glob import glob


class Dataloader:
    def __init__(self, image_folders, generator_func, limit=None):
        class_names = os.listdir(image_folders)

        for name in class_names:
            data = generator_func(image_folders, name, limit)
            np.save(os.path.join(image_folders, name, "data.npy"), data)



if __name__ == "__main__":
    Dataloader("Faces")