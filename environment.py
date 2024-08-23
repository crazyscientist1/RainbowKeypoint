class Variables():
<<<<<<< HEAD
    BASE_PATH = '/Users/kishorpallod/PycharmProjects/RainbowKeypoint/'
    INP_SIZE = (256, 256)
    OUT_SIZE = (64, 64)
    SIGMA = 2
    EPOCH_SIZE = 100
    LOSS = "Map matching" # "Map matching" or "UNET"
=======
    BASE_PATH = '/home/ubuntu/RainbowProject/RainbowKeypoint/'
    INP_SIZE = (256, 256)
    OUT_SIZE = (64, 64)
    SIGMA = 7
    EPOCH_SIZE = 100
    LOSS = "Map matching" # "Map matching" (output is keypoint heatmap) or "UNET" (single heatmap output)
    TENSOR_TYPE = "STRAIGHT" #"STRAIGHT" or "RAGGED"
>>>>>>> cc684ca (adding graph optimization)
