import os, cv2
from glob import glob
import pandas as pd
import tqdm
import numpy as np
import ast
import matplotlib.pyplot as plt
import tensorflow as tf
import ast
import os, cv2


def process_image(image_file):
    image_string = tf.io.read_file(image_file)
    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        return 0, image_string
    except tf.errors.InvalidArgumentError:
        return 1, image_string


def make_yolo_example_from_dataframe_row(df):
    # list의 형태로 담습니다.
    image_string = tf.io.read_file(df['img_path'])

    if isinstance(image_string, type(tf.constant(0))):  # tensor, eagertensor 이면 꺼낸다
        image_string = [image_string.numpy()]
    else:
        image_string = [image_string]

    # tf.example 생성
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'height':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[df['height']])),
            'width':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[df['width']])),
            'objectness':
            tf.train.Feature(int64_list=tf.train.Int64List(value=df["obj"])),
            'target_id':
            tf.train.Feature(int64_list=tf.train.Int64List(value=df["target_ids"])),
            'classes':
            tf.train.Feature(int64_list=tf.train.Int64List(value=df["classes"])),
            'x_min':
            tf.train.Feature(float_list=tf.train.FloatList(value=df['x_min'])),
            'y_min':
            tf.train.Feature(float_list=tf.train.FloatList(value=df['y_min'])),
            'x_max':
            tf.train.Feature(float_list=tf.train.FloatList(value=df['x_max'])),
            'y_max':
            tf.train.Feature(float_list=tf.train.FloatList(value=df['y_max'])), 
            'image_raw':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=image_string))
        }))
    return example

def activate_list_in_dataframe(df):
    for i in df.columns:
        a = df[i].values[0]
        if type(a) == type("a"):
            if a.startswith("["):
                df[i] = df[i].apply(lambda x: ast.literal_eval(x))
    return df

def make_dataset(anno_path, seq_path):
    """
    Make dataframe by annotation file

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
        "video_name", "frame_index", "shape", "target_id", "object_size", "score",
        "class", "truncation", "occlusion"
    ])

    v_name, frame, id, score, cls, trun, occ, shape, object_size, norm_object_size, x_min, x_max, x_cen, y_min, y_max, y_cen, bw, bh= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    g_anno = glob(anno_path + "/*")
    g_seq = glob(seq_path + "/*")

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
                temp_coco = list(map(int, line[2:6]))
                xmin = temp_coco[0]
                ymin = temp_coco[1]
                w = temp_coco[2]
                h = temp_coco[3]
                xmax = xmin + w
                ymax = ymin + h
                xcen = (xmin + xmax) / 2
                ycen = (ymin + ymax) / 2
                x_min.append(xmin)
                y_min.append(ymin)
                x_max.append(xmax)
                y_max.append(ymax)
                x_cen.append(xcen)
                y_cen.append(ycen)
                bw.append(w)
                bh.append(h)

                score.append(int(line[6]))
                cls.append(int(line[7]))
                trun.append(int(line[8]))
                occ.append(int(line[9]))
                shape.append(img_shape)
                object_size.append(temp_coco[2] * temp_coco[3])

    df["video_name"] = v_name
    df["frame_index"] = frame
    df["shape"] = shape
    df["target_id"] = id
    df["x_min"] = x_min
    df["y_min"] = y_min
    df["x_max"] = x_max
    df["y_max"] = y_max  
    df["x_cen"] = x_cen
    df["y_cen"] = y_cen
    df["bw"] = bw
    df["bh"] = bh 

    df["score"] = score
    df["class"] = cls
    df["truncation"] = trun
    df["occlusion"] = occ
    df["object_size"] = object_size
    df["unifed_class"] = df["class"].apply(lambda x: cls_unify(x))
    return df


def visualize_bbox_on_img(img, bbox, class_name, color=(255, 0, 0), thickness=2):

    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min +
                                                 w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color,
                  thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name,
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_bbox(imgs, bboxes, class_name):

    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    img = img[y_min:y:max, x_min:x_max]

    return img

def crop_box(imgs, boxes, pad=3):
    def get_crop_box(_img, _box, _pad):
        _box = ast.literal_eval(_box)
        x1, y1, w, h = _box
        x2,y2 = x1+w+_pad, y1+h+_pad
        x1, y1 = max(0, x1-_pad), max(0, y1-_pad)
        return _img[y1:y2, x1:x2]
    crops = [get_crop_box(img, box, pad) for (img, box) in zip(imgs, boxes)]
    return crops

def plot_ftus(crops, classname, n_cols=5, height_per_row=3):

    n_rows = int(np.ceil(len(crops)/n_cols)) 
    
    plt.figure(figsize=(18, n_rows*height_per_row)) 
    for i, crop in enumerate(crops): 
        plt.subplot(n_rows, n_cols, i+1) # subplot 생성과 각 image들의 자리 배치
        plt.imshow(crop)
        plt.title(f"{classname} #{i+1}  ––  SHAPE={crop.shape[:-1]}", fontweight="bold")
        plt.axis(False)
    plt.tight_layout()
    plt.show()

def get_sample_by_class(df, object_class, root_path, count=10, random_state=1, padding=3):

    """ get sample image """

    sample_df = df[df["class"] == object_class].sample(count, random_state=random_state)
    sample_df["img_path"] = root_path + sample_df["video_name"]+sample_df["frame_index"].apply(lambda x : "/"+str(x).zfill(7) +".jpg")
    imgs = [cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_RGB2BGR) for img_path in sample_df["img_path"].values]

    boxes = sample_df["coco"].values

    crops = crop_box(imgs, boxes, pad=padding)
    plot_ftus(crops, object_class)

def coco2xyxy(boxes):
    def _coco2xyxy(box):
        xmin,ymin,w,h = box
        xmax = xmin + w
        ymax = ymin + h
        return [xmin, ymin, xmax, ymax]
    return [_coco2xyxy(box) for box in boxes]

def coco2yolo(boxes):
    def _coco2yolo(box):
        xmin,ymin,w,h = box
        xmin = xmin + w/2
        ymin = ymin + h/2
        return [xmin, ymin, w, h]
    return [_coco2yolo(box) for box in boxes]

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
        heigth.append(ast.literal_eval(group["shape"].values[0])[0])
        target_ids.append(group["target_id"].values.tolist())
        width.append(ast.literal_eval(group["shape"].values[0])[1])
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
        img_path_list.append(df["video_name"].values[i] + "/" +str(df["frame_index"].values[i]).zfill(7) +".jpg")
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
    df["unified_classes"] = unified_classes
    return df

