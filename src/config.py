VOC_CLASSES = (  # always index 0
    "vehicle",
    "pedestrian",
    "cyclist",
)

VOC_IMG_MEAN = (80, 88, 97) # RGB

COLORS = [
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
]

# network expects a square input of this dimension
YOLO_IMG_DIM = (448, 448)
