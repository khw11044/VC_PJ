import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import cv2
import numpy as np
import argparse



parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--file', default='imgs/swing005_106.png', type=str,
                    help='videos or image files')
args = parser.parse_args()
file_name=args.file

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

IMG_SIZE = 480
THRESHOLD = 0.95

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

def output_model(file_name):
    img = cv2.imread(file_name) 
    img = cv2.resize(img, (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trf = T.Compose([
    T.ToTensor()
    ])
    input_img = trf(img)
    out = model([input_img])[0]

    return img, out

def HumanPoseEstimation(file_name):
    img, out = output_model(file_name)
    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().numpy()

        if score < THRESHOLD:
            continue

        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy() #[:, :2]
        neck = np.append((keypoints[6][:2] + (keypoints[5][:2] - keypoints[6][:2])/2),np.array([1]))
        neck[1] = neck.copy()[1] - 1*(neck.copy()[1] - keypoints[0][1])/5
        # 명치 추가 # 18번
        chest = np.append((keypoints[12][:2] + (keypoints[11][:2] - keypoints[12][:2])/2),np.array([1]))  # 11,12
        chest = neck.copy() - (neck.copy() - chest.copy())/2
        # 머리 추가 # 19번
        top = neck.copy()
        top[1] = keypoints[0][1] - (neck[1] - keypoints[0][1]) * 1
        new_keypoints = np.empty((0,3), dtype=float)
        new_keypoints = np.append(new_keypoints, np.array([top,keypoints[0],neck,chest,
                                keypoints[6],keypoints[8],keypoints[10],
                                keypoints[5],keypoints[7],keypoints[9],
                                keypoints[11],keypoints[13],keypoints[15],
                                keypoints[12],keypoints[14],keypoints[16]
                                ]), axis=0)

        cv2.rectangle(img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), thickness=2, color=(0, 0, 255))

        for k in new_keypoints:
            print(k[:2])
            if k[2] == 1:
                cv2.circle(img, center=tuple(k[:2].astype(int)), radius=4, color=(255, 0, 0), thickness=-1)
            else :
                cv2.circle(img, center=tuple(k[:2].astype(int)), radius=4, color=(0, 255, 155), thickness=1)

        cv2.polylines(img, pts=[new_keypoints[0:4,:2].astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(new_keypoints[4:7,:2],0,new_keypoints[2,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(new_keypoints[7:10,:2],0,new_keypoints[2,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(new_keypoints[10:13,:2],0,new_keypoints[3,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(new_keypoints[13:16,:2],0,new_keypoints[3,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite('test.png',img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    cv2.waitKey(0)

if __name__ == "__main__":
    HumanPoseEstimation(file_name)

