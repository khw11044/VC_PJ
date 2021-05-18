
# 폴더를 지정하고 거기에 있는 폴더를 모두 조회 
# argu로 +이름을 정함 예: seg or pose 

import os 
import json
import argparse

parser = argparse.ArgumentParser(description="label_folder")
parser.add_argument('--name', default='./label/pose/', type=str,
                    help='label_folder')
args = parser.parse_args()
root=args.name


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

# swing001 폴더안에 json파일 모두 읽고 하나로 통합
def dataLoader_img(root_dir):
    print(root_dir)
    informs = dict()
    
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file in files:
                img_name = str(file.split('_pose')[0] + ".png")
                json_name = str(root) +'/' + str(file)
                with open(json_name, 'r') as f:
                    json_file = json.load(f)
                    informs[img_name] = json_file


    json_file_name = root_dir.split('/')[-1] + '.json'
    with open(json_file_name, 'w', encoding='utf-8') as make_file:
        json.dump(informs, make_file, indent="\t")

# pose 폴더에 모든 swing폴더 모두 읽고 각swing별 json파일 통합
def load_all_folder(root_dir):
    for (root, dirs, files) in os.walk(root_dir):
        if len(dirs) > 0 :
            for dir in dirs:
                print(root_dir + '/' + dir)
                dataLoader_img(root_dir + '/' + dir)

    

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        print(prefix + path)


if __name__ == "__main__":

    # dataLoader_img(root)
    load_all_folder(root)
