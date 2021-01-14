import cv2

def prepare_by_stretch_img(img, img_size):
    shape = (img.shape[1], img.shape[2])

    if shape[0] < img_size[0] or shape[1] < img_size[1]:
        coef_0 = float(img_size[0]) / shape[0]
        coef_1 = float(img_size[1]) / shape[1]

        coef = int(max(coef_0, coef_1)) + 1

        return cv2.resize(img, (shape[0]*coef, shape[1]*coef))
        
    return img

def prepare_by_reshape(img, img_size):
    return cv2.resize(img, (img_size[0], img_size[1]))

def prepare_none(img, img_size):
    return img

def get_prepare_function(prepare_function_name):
    d = {
        'none': prepare_none,
        'reshape': prepare_by_reshape,
        'stretch': prepare_by_stretch_img,
    }
    return d[prepare_function_name]