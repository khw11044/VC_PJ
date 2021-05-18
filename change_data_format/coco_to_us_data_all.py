'''
여러 이미지 정보가 들어있는 coco파일 하나를 읽고 여러 이미지에 대한 pose.json과 bb.json의 우리 데이터 format으로 배출
'''

import json 
from collections import OrderedDict
import argparse



parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--file', default='label_input/output.json', type=str,
                    help='videos or image files')
args = parser.parse_args()
root=args.file

def read_json(root):
    with open(root,'r') as f:
        json_data = json.load(f)
    
    # print(json.dumps(json_data))
    return json_data

def export_json(file):       # [name,[new_keypoints],box]
    label_data = OrderedDict()
    label_data2 = OrderedDict()

    json_data = read_json(file)
    for i in range(len(json_data["images"])):
        image = json_data["images"][i]
        annotation = json_data["annotations"][i]
        image_id = image["id"]

        file_name = image["file_name"]
        keypoint_index = annotation["keypoints"]
        bb = annotation["bbox"]
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

    print('export json_label')
    json_file_name = json_data["images"][0]["file_name"].split('.')[0].split('_')[0]
    print("json_file_name",json_file_name)
    with open('label_output/' + json_file_name +'_pose.json', 'w', encoding="utf-8") as make_file1:
        json.dump(label_data, make_file1, ensure_ascii=False, indent="\t")

    print('export json_label2')
    with open('label_output/' + json_file_name +'_bb.json', 'w', encoding="utf-8") as make_file2:
        json.dump(label_data2, make_file2, ensure_ascii=False, indent="\t")


if __name__ == "__main__":
    # root = "output.json"
    export_json(root)
    # print(len(json_data["images"]))