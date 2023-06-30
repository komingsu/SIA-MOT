import os, cv2
from glob import glob
import pandas as pd
from tqdm.auto import tqdm

def make_dataset(anno_path, seq_path):
    """
    input : 
    anno_path : annotation txt 파일 들이있는 폴더 path
    seq_path  : video folder 가 들어있는 폴더 path

    output :
    dataframe
    """

    def cls_unify(x):
        if x == 0:
            return 0
        elif x <= 2:
            return 1
        else:
            return 2

    df = pd.DataFrame(columns=[
        "video_name", "frame_index", "shape", "target_id", "xywh", "norm_xywh",
        "object_size", "norm_object_size", "score", "class", "truncation",
        "occlusion"
    ])

    v_name, frame, id, xywh, score, cls, trun, occ, shape, object_size, norm_xywh, norm_object_size = [],[],[],[],[],[],[],[],[],[],[],[]

    g_anno = glob(anno_path + "/*").sort()
    g_seq = glob(seq_path + "/*").sort()

    for anno, seq in tqdm(zip(g_anno, g_seq)):
        base, name = os.path.split(anno)

        name, ex = os.path.splitext(name)
        name, ex = os.path.splitext(name)
        base, _ = os.path.split(base)

        img = cv2.imread(seq + "/0000001.jpg")
        img_shape = img.shape

        with open(anno, "r") as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.split(",")
                v_name.append(name)
                frame.append(int(line[0]))
                id.append(int(line[1]))
                temp_xywh = list(map(int, line[2:6]))
                xywh.append(temp_xywh)
                nx = temp_xywh[0] / img_shape[1]
                ny = temp_xywh[1] / img_shape[0]
                nw = temp_xywh[2] / img_shape[1]
                nh = temp_xywh[3] / img_shape[0]
                norm_xywh.append([nx, ny, nw, nh])

                score.append(int(line[6]))
                cls.append(int(line[7]))
                trun.append(int(line[8]))
                occ.append(int(line[9]))
                shape.append(img_shape)
                object_size.append(temp_xywh[2] * temp_xywh[3])
                norm_object_size.append(nw * nh)

    df["video_name"] = v_name
    df["frame_index"] = frame
    df["shape"] = shape
    df["target_id"] = id
    df["xywh"] = xywh
    df["norm_xywh"] = norm_xywh
    df["score"] = score
    df["class"] = cls
    df["truncation"] = trun
    df["occlusion"] = occ
    df["object_size"] = object_size
    df["norm_object_size"] = norm_object_size
    df["unifed_class"] = df["class"].apply(lambda x: cls_unify(x))
    return df