import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import cv2
import numpy as np

from exportJson import export_json

import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--folder', default="./imgs", type=str,
                    help='videos or image files')
args = parser.parse_args()
root=args.folder

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

IMG_SIZE = 480
THRESHOLD = 0.95
folder_all_image = []

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()



def output_model(file_name):
    img = cv2.imread(file_name) 
    # img = cv2.resize(img, (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
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
        # 왼쪽 발끝, 발뒷꿈치

        # 오른쪽 발끝, 발뒷꿈치 

        # Golf club head

        keypoints = np.array([top.astype(int),keypoints[0].astype(int),neck.astype(int),chest.astype(int), #TopHead,Nose,Neck,Chest,
                                keypoints[6].astype(int),keypoints[8].astype(int),keypoints[10].astype(int),                            #R_Shol,R_El,R_Wr
                                keypoints[5].astype(int),keypoints[7].astype(int),keypoints[9].astype(int),                             #L_Shol,L_El,L_Wr                 
                                keypoints[11].astype(int),keypoints[13].astype(int),keypoints[15].astype(int),                          #R_Hip,R_kn,R_An
                                keypoints[12].astype(int),keypoints[14].astype(int),keypoints[16].astype(int)                           #L_Hip,L_kn,L_An
                                ])

        # ---- drawing ------
        cv2.rectangle(img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), thickness=2, color=(0, 0, 255))

        for k in keypoints:
            if k[2] == 1:
                cv2.circle(img, center=tuple(k[:2].astype(int)), radius=8, color=(255, 0, 0), thickness=-1)
            else :
                cv2.circle(img, center=tuple(k[:2].astype(int)), radius=8, color=(0, 255, 155), thickness=1)

        cv2.polylines(img, pts=[keypoints[0:4,:2].astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(keypoints[4:7,:2],0,keypoints[2,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(keypoints[7:10,:2],0,keypoints[2,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(keypoints[10:13,:2],0,keypoints[3,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, pts=[np.insert(keypoints[13:16,:2],0,keypoints[3,:2],axis=0).astype(int)], isClosed=False, color=(0, 255, 0), thickness=2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img,keypoints.tolist(),box.tolist()

# 폴더 이미지 모두 읽어 오기 
def get_images(root):
    images = glob(os.path.join(root, '*png'))
    images = sorted(images, key=lambda x: x.split('/')[-1].split('.')[0])
    print(images)

    for image in images:
        img,new_keypoints,box = HumanPoseEstimation(image)
        box = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
        name = image.split('\\')[-1]
        save_name = "keypoint_images/"+str(name.split('.')[0]) + "_pose.png"
        folder_all_image.append([name,new_keypoints,box])    # [id,[x,y,v],]
        resize_image = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_name,resize_image)
        yield resize_image
    export_json(folder_all_image)


# 사용자에게 완전히 보여주기 용
def video(root):
    name = root.split('/')[-1].split('.')[0]
    for frame in get_images(root):
        
        cv2.imshow(name, frame)
        cv2.waitKey(40)

if __name__ == "__main__":
    video(root)

