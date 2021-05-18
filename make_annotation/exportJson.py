import json 
from collections import OrderedDict

def export_json(folder_all_pose):       # [name,[new_keypoints],box]
    label_data = OrderedDict()
    informs = dict()
    shape = dict()
    top = 0
    dots = []
    # regions attribute
    for file_data in folder_all_pose:            # [filename, [[id,[x,y,z]],[id,[x,y,z]].....]]
        file_name, pose,box = file_data[0], file_data[-2],file_data[-1]   # [[id,[x,y,v]],[id,[x,y,v]].....]

        informs = {"person_count":1, "image_id":top, "image_name":file_name}

        shape = {"keypoint_index": pose, "box":box ,"image_id":top}
        top += 1

        img_dict = [ 
                    "info", informs,
                    "key", shape
                    ]       

        label_data[file_name] = img_dict

    print('export json_label')
    with open('label_data.json', 'w', encoding="utf-8") as make_file:
        json.dump(label_data, make_file, ensure_ascii=False, indent="\t")