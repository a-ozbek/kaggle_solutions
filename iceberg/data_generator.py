import numpy as np
from skimage import io, transform
from keras.applications import xception
import itertools
import random
RANDOM_SEED = 43
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def class2dummy(classes, class_label):
    return [1 if c==class_label else 0 for c in classes]


def augment_im(im):
    # Functions
    def bypass(im):
        return im
    
    def flip_h(im):
        return im[:, ::-1, :]
    
    def rotate_im(im, max_angle=10):
        rot_angle = random.randint(-1 * max_angle, max_angle)
        im = transform.rotate(im, angle=rot_angle, mode='symmetric', preserve_range=True).astype(np.uint8)
        return im
    
    def zoom_im(im, factor=0.33):
        factor = np.clip(np.random.rand() * factor, a_min=0.1, a_max=None)
        return zoom_im_helper(im, factor)

    def zoom_im_helper(im, factor):
        original_size = im.shape
        half_height, half_width = original_size[0] / 2, original_size[0] / 2    
        height_crop = int(half_height * factor)
        width_crop = int(half_width * factor)
        im_zoom = im[height_crop:original_size[0]-height_crop, 
                     width_crop:original_size[1]-width_crop, 
                     :]
        im_zoom = (transform.resize(im_zoom, output_shape=original_size) * 255.0).astype(np.uint8)
        return im_zoom
        
    # Apply functions
    functions = [bypass, flip_h, rotate_im]
    functions = random.sample(functions, random.randint(1, len(functions)))    
    for function in functions:
        im = function(im)
    
    return im  



def data_generator(data, classes, steps_per_epoch, batch_size, image_size, augment=False):
    """
    Data generator
    
    data: List of filenames and their labels
    """    
    X, y = [], []
    data = list(np.copy(data))
    random.shuffle(data)
    total_steps_elapsed = 0
    while True:
        iter_cycle_obj = itertools.cycle(data)
        for im_filename, label in iter_cycle_obj:
            # Get image
            im = io.imread(im_filename)
            if augment:
                im = augment_im(im)
            im = transform.resize(im, output_shape=image_size)
            # Get label
            label = class2dummy(classes, label)
            # Append X, y
            X.append(im)
            y.append(label)        
            # Check if batch_size is reached
            if len(y) > batch_size - 1:
                X, y = np.array(X), np.array(y)
                X = xception.preprocess_input(X * 255.0)
                yield (X, y)
                X, y = [], []
                total_steps_elapsed += 1            
            # Check total_steps_elapsed (end of epoch)            
            if total_steps_elapsed > steps_per_epoch:
                random.shuffle(data)
                total_steps_elapsed = 0  
                break