'''
하나짜리 coco파일 하나만 읽고 하나의 이미지에 대한 pose.json과 bb.json의 우리 데이터 format으로 배출
'''

import json 
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--file', default='label_input/testset.json', type=str,
                    help='videos or image files')
args = parser.parse_args()
root=args.file

def read_json(file):
    with open(file,'r') as f:
        json_data = json.load(f)

    return json_data

def export_json(file):       
    label_data = OrderedDict()
    label_data2 = OrderedDict()

    # 필요한 정보만 뽑기
    json_data = read_json(file)
    image_id = json_data["images"][0]["id"]
    file_name = json_data["images"][0]["file_name"]
    keypoint_index = json_data["annotations"][0]["keypoints"]
    bb = json_data["annotations"][0]["bbox"]


    # pose.json 형식
    pose_informs = {"person_count":1, "image_id":image_id, "image_name":file_name}
    pose_key = {"image_id":image_id, "keypoint_index": keypoint_index}
    # bb.json 형식
    bb_informs = {"image_name":file_name, "image_id":str(image_id), "person_count":1, "club_count":1,"ball_count":1 }
    bb_bb = {"image_id":image_id, "person_bb": bb,"club_bb": [],"ball_bb": []}
    pose_data = [ 
                "info", pose_informs,
                "key", pose_key
                ]       

    bb_data = [{"info":bb_informs, "bb":bb_bb}] 

    label_data[file_name] = pose_data
    label_data2[file_name] = bb_data

    json_file_name = json_data["images"][0]["file_name"].split('.')[0].split('_')[0]
    print('export json_label')
    with open('label_output/' + json_file_name +'_pose_one.json', 'w', encoding="utf-8") as make_file1:
        json.dump(label_data, make_file1, ensure_ascii=False, indent="\t")

    print('export json_label2')
    with open('label_output/' + json_file_name +'_bb_one.json', 'w', encoding="utf-8") as make_file2:
        json.dump(label_data2, make_file2, ensure_ascii=False, indent="\t")


if __name__ == "__main__":

    export_json(root)
    