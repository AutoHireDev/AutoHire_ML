from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
import PIL.Image as Image
import cv2
import os
import glob
import numpy as np


data = torch.load("data.pt")
faces, names = data[0], data[1]
mtcnn = MTCNN()
res = InceptionResnetV1(pretrained="vggface2").eval()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# res=torch.load("flayer_model.pt",map_location=torch.device('cpu'))
res = res.to(device)


class Facerec:
    def __init__(self):
        self.known_face_encodings = faces
        self.known_face_names = names

        self.frame_resizing = 1

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            print(filename)
            print(type(res))
            img, p = mtcnn(rgb_img, return_prob=True)
            if type(img) != type(None):
                img = img.to(device)
                img_1 = res(img.unsqueeze(0))

                self.known_face_encodings.append(img_1.detach())
                self.known_face_names.append(filename)
            else:
                return f"Unable to detect the {filename}"
        data = [self.known_face_encodings, self.known_face_names]
        torch.save(data, "data.pt")
        return True

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing
        )

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = mtcnn.detect(rgb_small_frame)

        if type(face_locations[0]) != type(None):
            face = mtcnn(rgb_small_frame)
            face.to(device)
            face_1 = res(face.unsqueeze(0).to(device))
            face_locations = face_locations[0][0]
            face_names = []
            face_acc = []
            if len(self.known_face_encodings) == 0:
                return 0
            else:
                dist_list = []
                for idx, emb in enumerate(self.known_face_encodings):
                    dist = torch.dist(face_1, emb).item()
                    dist_list.append(dist)
                idx_min = dist_list.index(min(dist_list))
                if min(dist_list) < 0.8:
                    return 1
                else:
                    return 0
        else:
            return 0

    def clear_encodings():
        faces = []
        names = []
        d = [faces, names]
        torch.save(d, "data.pt")
        return True
