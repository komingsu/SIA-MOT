# import albumentations as A
import cv2

DATAPATH = "D:visdrone/"
IMAGE_SIZE = 416
class_change_dict = {
    1:0,#"pedestrian"
    2:0,#"people"
    3:1,#"bicycle"
    4:2,#"car"
    5:2,#"van"
    6:2,#"truck"
    7:1,#"tricycle"
    8:1,#"awning-tricycle"
    9:2,# "bus"
    10:1,#"motor"
    }


DATA_FORMAT = "pascal_voc"

# scale = 1.1
# train_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
#         A.PadIfNeeded(
#             min_height=int(IMAGE_SIZE * scale),
#             min_width=int(IMAGE_SIZE * scale),
#             border_mode=cv2.BORDER_CONSTANT,
#         ),
#         A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
#         A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
#         A.OneOf(
#             [
#                 A.ShiftScaleRotate(
#                     rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
#                 ),
#                 A.IAAAffine(shear=15, p=0.5, mode="constant"),
#             ],
#             p=1.0,
#         ),
#         A.HorizontalFlip(p=0.5),
#         A.Blur(p=0.1),
#         A.CLAHE(p=0.1),
#         A.Posterize(p=0.1),
#         A.ToGray(p=0.1),
#         A.ChannelShuffle(p=0.05),
#     ],
#     bbox_params=A.BboxParams(format=DATA_FORMAT, min_visibility=0.4, label_fields=[],),
# )
# test_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=IMAGE_SIZE),
#         A.PadIfNeeded(
#             min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
#         ),
#     ],
#     bbox_params=A.BboxParams(format=DATA_FORMAT, min_visibility=0.4, label_fields=[]),
# )
