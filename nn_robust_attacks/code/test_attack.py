## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
import time
from PIL import Image
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel
from keras import backend as K
from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if Image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])
    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        # data, model =  MNIST(), MNISTModel("models/mnist", sess)
        data, model = CIFAR(), CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        # attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)
        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")

        for i in range(len(adv)):
            print("Valid:")
            # show(inputs[i])
            array_to_img(inputs[i]).save(f'clean{i}.png')
            print("Adversarial:")
            # show(adv[i])
            array_to_img(adv[i]).save(f'adv{i}.png')
            print("Classification:", np.argmax(model.model.predict(adv[i:i + 1])))

            print("Total distortion:", np.sum((adv[i] - inputs[i]) ** 2) ** .5)
