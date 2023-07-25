import tensorflow as tf
import os

dataset_name ="custom_visdrone"

if dataset_name == "custom_visdrone":
    data_dir = './EDA/'
    classes  = './EDA/visdrone.names'
    
    model_ckpt_dir = './checkpoints/basemodel_try_01/'
    
model_ckpt = os.path.join(model_ckpt_dir,'yolov3.tf')
tf_record_train = os.path.join(data_dir, 'train_visdrone.tfrecord')
tf_record_val   = os.path.join(data_dir, 'val_visdrone.tfrecord')
tf_record_test   = os.path.join(data_dir, 'test_visdrone.tfrecord')

# load data from local
raw_train_dataset = tf.data.TFRecordDataset(tf_record_train)
raw_val_dataset   = tf.data.TFRecordDataset(tf_record_val)
raw_test_dataset   = tf.data.TFRecordDataset(tf_record_test)
class_names = [c.strip() for c in open(classes).readlines()]

# Common for all datasets
n_classes = len(class_names)
# print(len(class_names))