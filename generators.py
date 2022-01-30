import numpy as np
import os
from glob import glob
import cv2
from autocrop import Cropper
from PIL import Image
from tqdm import tqdm


def create_images_from_video(image_folder, name, limit=200):
    path = os.path.join(image_folder, name, "Video.mp4")
    assert os.path.exists(path), "Should be structured by /{image_folder}/{name}/Video.mp4"

    os.mkdir(os.path.join(image_folder, name, "Images"))
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        savename = os.path.join(image_folder, name, f"Images/{count:05d}.png")
        cv2.imwrite(savename, image)
        success, image = vidcap.read()
        count += 1

        if count > limit:
            break

    print(f"Created Images for {name}. Check the folder. Proceed with creating cropped face")


def create_crop_faces(image_folder, name, height=256, width=256, face_percent=70):
    assert os.path.exists(os.path.join(image_folder, name,
                                       "Images")), "folder structure is /{image_folder}/{name}/{Images}/ - make sure you generate from video first or dowload your own images"

    files = glob(os.path.join(image_folder, name, "Images", "*.png"))
    cropper = Cropper(height=height, width=width, face_percent=face_percent)


    output_path = os.path.join(image_folder, name, "Output")
    os.mkdir(output_path)
    for i, file in tqdm(enumerate(files)):
        crop = cropper.crop(file)
        if crop is None:
            continue

        image = Image.fromarray(crop)
        image.save(os.path.join(output_path, f"{i}.png"))


if __name__ == "__main__":
    create_crop_faces("Faces", "Ludwig")
