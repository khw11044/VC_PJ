
# 폴더를 지정하고 거기에 있는 폴더를 모두 조회 
# argu로 +이름을 정함 예: seg or pose 

import os 
import json
import argparse

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()
root=args.video_name


def listdirLoader(root): 
    files = []
    #root = '../siamfc-pytorch/tools/data/NonVideo4_tiny'
    path = os.listdir(root)
    return path


def folder_rename(folder_list,new_folder_name):
    for folder in folder_list:
        print(folder)
        old_name = os.path.join(root, folder)
        new_name = os.path.join(root, str(folder.split('_')[-1]))
        print(old_name)
        print(new_name)
        os.rename(old_name, new_name)


def dataLoader_img(root_dir):
    informs = dict()
    
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file in files:
                img_name = str(file.split('_pose')[0] + ".png")
                json_name = str(root) +'/' + str(file)
                with open(json_name, 'r') as f:
                    json_file = json.load(f)
                    informs[img_name] = json_file


    print(root_dir)
    # with open('./test.json', 'w', encoding='utf-8') as make_file:
    #     json.dump(informs, make_file, indent="\t")

                
    

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        print(prefix + path)


if __name__ == "__main__":

    dataLoader_img(root)
