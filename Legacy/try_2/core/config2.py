from easydict import EasyDict as edict


__C                           = edict()

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.CLASSES              = "./data/classes/custom_visdrone.names"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/visdrone_train.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 1
__C.TRAIN.EPOCHS              = 10

# VALID options
__C.VALID                     = edict()

__C.VALID.ANNOT_PATH          = "./data/dataset/visdrone_val.txt"
__C.VALID.BATCH_SIZE          = 4
# __C.VALID.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.VALID.INPUT_SIZE          = [416]
__C.VALID.DATA_AUG            = False


# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/visdrone_test.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = [416]
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.2
__C.TEST.IOU_THRESHOLD        = 0.2


