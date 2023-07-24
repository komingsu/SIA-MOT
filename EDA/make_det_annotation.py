import os, cv2
import glob
import pandas as pd
from tqdm.auto import tqdm
import config
"""
annotation txt 파일을 생성합니다.

목표
train, val, test의 파일을 생성
img_path, boxes[x_min, y_min, x_max, y_max , class]들을 나열

방법
1. 각 이미지별 annotation들을 읽

"""


def make_dataset(anno_path, seq_path):
    """
    Make dataframe by annotation file

    input : 
    anno_path : annotation txt 파일 들이있는 폴더 path
    seq_path  : video folder 가 들어있는 폴더 path

    output :
    dataframe
    """

    df = pd.DataFrame()
    v_name= []
    frame= []
    obj_id= []
    score= []
    classes= []
    trun= []
    occ= []
    shape= []
    object_size= []
    x_min= []
    x_max= []
    x_cen= []
    y_min= []
    y_max= []
    y_cen= []
    bw= []
    bh= []
    g_anno = sorted(glob.glob(anno_path + "/*"))
    g_seq = sorted(glob.glob(seq_path + "/*"))
    for anno, seq in tqdm(zip(g_anno, g_seq)):
        # 예시
        # anno    :"D:/visdrone/VisDrone2019-MOT-test/annotations/uav0000009_03358_v.txt"
        # seq     :"D:/visdrone/VisDrone2019-MOT-test/sequences/uav0000009_03358_v"
        base, txtfile = os.path.split(anno)
        # base    :"D:/visdrone/VisDrone2019-MOT-test/annotations"
        # txtfile :"uav0000009_03358_v.txt"
        name, ex = os.path.splitext(txtfile)
        # name    :"uav0000013_00000_v"
        # ex      :".txt"
        img = cv2.imread(seq + f"/{name}_0000001.jpg")
        height = img.shape[0]
        width = img.shape[1]

        with open(anno, "r") as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.split(",")
                v_name.append(name)
                frame.append(int(line[0]))
                obj_id.append(int(line[1]))
                temp_coco = list(map(int, line[2:6]))
                xmin = temp_coco[0]
                ymin = temp_coco[1]
                w = temp_coco[2]
                h = temp_coco[3]

                xmax = np.clip(xmin + w,1, width-1)
                ymax = np.clip(ymin + h,1, height-1)
                xcen = (xmin + xmax) / 2
                ycen = (ymin + ymax) / 2
                w = xmax-xmin
                h = ymax-ymin

                x_min.append(xmin)
                y_min.append(ymin)
                x_max.append(xmax)
                y_max.append(ymax)
                x_cen.append(xcen)
                y_cen.append(ycen)
                bw.append(w)
                bh.append(h)

                score.append(int(line[6]))
                classes.append(int(line[7]))
                trun.append(int(line[8]))
                occ.append(int(line[9]))
                object_size.append(w * h)

    df["video_name"] = v_name
    df["frame_index"] = frame
    df["target_id"] = obj_id
    df["x_min"] = x_min
    df["y_min"] = y_min
    df["x_max"] = x_max
    df["y_max"] = y_max  
    df["x_cen"] = x_cen
    df["y_cen"] = y_cen
    df["bw"] = bw
    df["bh"] = bh 

    df["score"] = score
    df["class"] = classes
    df["truncation"] = trun
    df["occlusion"] = occ
    df["object_size"] = object_size

    return df

def make_detection_dataframe(data):
    target_ids = []
    video_name = []
    frame_index = []
    heigth = []
    width = []
    coco  = []
    classes = []
    obj=[]
    unified_classes = []
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    x_cen = []
    y_cen = []
    b_width = []
    b_height = []

    for (k1,k2), group in data.groupby(['video_name',"frame_index"]):
        video_name.append(k1)
        frame_index.append(k2)
        heigth.append(group["shape"].values[0][0])
        target_ids.append(group["target_id"].values.tolist())
        width.append(group["shape"].values[0][1])
        x_min.append( group["x_min"].values.tolist() )
        y_min.append( group["y_min"].values.tolist() )
        x_max.append( group["x_max"].values.tolist() )
        y_max.append( group["y_max"].values.tolist() )
        x_cen.append( group["x_cen"].values.tolist() )
        y_cen.append( group["y_cen"].values.tolist() )
        b_width.append( group["bw"].values.tolist() )
        b_height.append( group["bh"].values.tolist() )
        classes.append( group["class"].values.tolist() )
        obj.append( group["score"].values.tolist() )
        unified_classes.append( group["unifed_class"].values.tolist())
    df = pd.DataFrame()
    df["video_name"] = video_name
    df["frame_index"] = frame_index
    
    img_path_list = []
    for i in range(len(df)):
        img_path_list.append(df["video_name"].values[i] + "/"+df["video_name"].values[i]+"_"+str(df["frame_index"].values[i]).zfill(7) +".jpg")
    df["img_path"] = img_path_list
    df["target_ids"] = target_ids
    df["x_min"] = x_min
    df["x_max"] = x_max
    df["y_min"] = y_min
    df["y_max"] = y_max
    df["x_cen"] = x_cen
    df["y_cen"] = y_cen
    df["b_width"] = b_width
    df["b_height"] = b_height

    df["height"] = heigth
    df["width"] = width
    df["classes"] = classes
    df["obj"] = obj
    return df

if __name__ =="__main__":
    for dtype in ["train","val","test"]: 
        anno_path = os.path.join(config.DATAPATH,f"VisDrone2019-MOT-{dtype}","annotations")
        seq_path = os.path.join(config.DATAPATH,f"VisDrone2019-MOT-{dtype}" ,"sequences")
        df = make_dataset(anno_path, seq_path)
        df.to_csv(f"{dtype}_dataframe.csv",index=False)
        df = make_detection_dataframe(df)
        df.to_csv(f"{dtype}_detection_frame.csv",index=False)