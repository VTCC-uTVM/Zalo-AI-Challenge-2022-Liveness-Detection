import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageOps
import albumentations.augmentations.functional as func

from albumentations import (
    Rotate, RandomResizedCrop, GridDistortion, OpticalDistortion, IAAPerspective, ElasticTransform
)
import inspect
import cv2


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.
input_image = 'result1.jpg'
images_folder = 'images'


def cutout(img, pad_size, replace=127):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
        img (Tensor): Tensor image of size (C, H, W).
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.

    """
    n_holes = 1
    new_img = img.copy()
    h, w, _ = new_img.shape

    #mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - pad_size, 0, h)
        y2 = np.clip(y + pad_size, 0, h)
        x1 = np.clip(x - pad_size, 0, w)
        x2 = np.clip(x + pad_size, 0, w)
        new_img[y1: y2, x1: x2] = replace

    return new_img

def contrast(image, factor):
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    img = np.asarray(enhancer.enhance(factor))
    return img


def autocontrast(image):
    pil_image = Image.fromarray(image)
    return np.asarray(ImageOps.autocontrast(pil_image))


def color(image, factor):
    pil_image = Image.fromarray(image)
    return np.asarray(ImageEnhance.Color(pil_image).enhance(factor))


def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return func.solarize(image, threshold)


def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = addition + image.astype(np.int64)
    added_image = np.clip(added_image, 0, 255).astype(np.uint8)
    return np.where(image < threshold, added_image, image)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    pil_image = Image.fromarray(image)
    return np.asarray(ImageEnhance.Brightness(pil_image).enhance(factor))


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    return func.posterize(image, bits)


def rotate(image, degrees, replace):
    """Rotates the image by degrees either clockwise or counterclockwise.
    Args:
      image: An PIL image
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tensor to fill empty pixels caused by
        the rotate operation.
    Returns:
      The rotated version of image.
    """
    image = Rotate(degrees, p=1)(image=image)['image']
    return image


def shear_x(image, level, replace):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    level = float(level)
    rows, cols, _ = image.shape
    matrix = np.float32([[1., level, 0],[0., 1., 0.]])
    
    image = cv2.warpAffine(image,matrix,(cols,rows), borderValue=replace)
    return image


def shear_y(image, level, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    level = float(level)
    rows, cols, _ = image.shape
    
    matrix = np.float32([[1., 0., 0],[level, 1., 0.]])
    
    image = cv2.warpAffine(image,matrix,(cols,rows), borderValue=replace)
    return image


def translate_x(image, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    pixels = float(pixels)
    rows, cols, _ = image.shape
    matrix = np.float32([[1., 0., -pixels],[0., 1., 0.]])
    image = cv2.warpAffine(image,matrix,(cols,rows), borderValue=replace)
    return image


def translate_y(image, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    pixels = float(pixels)
    rows, cols, _ = image.shape
    matrix = np.float32([[1., 0., 0],[0., 1., -pixels]])
    image = cv2.warpAffine(image,matrix,(cols,rows), borderValue=replace)
    return image


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    pil_image = Image.fromarray(image)
    return np.asarray(ImageEnhance.Sharpness(pil_image).enhance(factor))


def equalize(image):
    pil_image = Image.fromarray(image)
    return np.asarray(ImageOps.equalize(pil_image))


def invert(image):
    """Inverts the image pixels."""
    pil_image = Image.fromarray(image)
    return np.asarray(ImageOps.invert(pil_image))


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout
}


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = bool(np.floor(np.random.uniform() + 0.5))
    # print(np.random.uniform())

    if should_flip:
        return tensor
    else:
        return -tensor


def _rotate_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return (level,)


def _enhance_level_to_arg(level):
    return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _translate_level_to_arg(level, translate_const):
    level = (level/_MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    #print(level)
    return (level,)


def level_to_arg(hparams):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Rotate': _rotate_level_to_arg,
        'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': lambda level: (int((level/_MAX_LEVEL) * hparams['cutout_const']),),
        # pylint:disable=g-long-lambda
        'TranslateX': lambda level: _translate_level_to_arg(
            level, hparams['translate_const']),
        'TranslateY': lambda level: _translate_level_to_arg(
            level, hparams['translate_const']),
        # pylint:enable=g-long-lambda
    }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(augmentation_hparams)[name](level)

    # Check to see if prob is passed into function. This is used for operations
    # where we alter bboxes independently.
    # pytype:disable=wrong-arg-types
    if 'prob' in inspect.getargspec(func)[0]:
        args = tuple([prob] + list(args))
    # pytype:enable=wrong-arg-types

    # Add in replace arg if it is required for the function that is being called.
    if 'replace' in inspect.getargspec(func)[0]:
        # Make sure replace is the final argument
        assert 'replace' == inspect.getargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])

    return (func, prob, args)


def distort_image_with_randaugment(image, n_frames, num_layers, magnitude):
    """Applies the RandAugment policy to `image`.
    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 30].
    Returns:
      The augmented version of `image`.
    """
    replace_value = [128] * 3
    #print('Using RandAug.')
    augmentation_hparams = {
        "cutout_const": 40, "translate_const": 100}
    available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
        'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'SolarizeAdd', 'Cutout']

    for layer_num in range(num_layers):
        op_to_select = np.random.randint(0, len(available_ops))
        random_magnitude = float(magnitude)

        for (i, op_name) in enumerate(available_ops):
            prob = np.random.uniform(0.2, 0.8)
            func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                               replace_value, augmentation_hparams)
            if i == op_to_select:
                #print("Op select: ", available_ops[op_to_select])
                image = func(image, *args)

    return image


def preprocess_input(image, n_frames, randaug_num_layers = 1, randaug_magnitude = 9):
    return distort_image_with_randaugment(
        image, n_frames, randaug_num_layers, randaug_magnitude)


def load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def test_cutout():
    img = load_image(input_image)

    for i in range(3):
        aug = cutout(img, 40)
        plt.imshow(aug)
        plt.show()
    aug.save(os.path.join(images_folder, "cutout.jpg"))


def test_contrast():
    img = load_image(input_image)
    aug = contrast(img, 1.3)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "contrast.jpg"))


def test_autocontrast():
    img = load_image(input_image)
    aug = autocontrast(img)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "autocontrast.jpg"))


def test_color():
    img = load_image(input_image)
    aug = color(img, 0.3)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "color.jpg"))


def test_solarize():
    img = load_image(input_image)
    aug = solarize(img, 128)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "solarize.jpg"))


def test_solarize_add():
    img = load_image(input_image)
    aug = solarize_add(img, 50, 128)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "solarize_add.jpg"))


def test_brightness():
    img = load_image(input_image)
    aug = brightness(img, 1.8)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "brightness.jpg"))


def test_posterize():
    img = load_image(input_image)
    aug = posterize(img, 3)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "posterize.jpg"))


def test_rotate():
    img = load_image(input_image)
    aug = rotate(img, 25, [128]*3)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "rotate.jpg"))


def test_shear():
    img = load_image(input_image)
    aug_shearx = shear_x(img, 0.2, [128]*3)
    plt.imshow(aug_shearx)
    plt.show()
    aug_shearx.save(os.path.join(images_folder, "shear_x.jpg"))
    aug_sheary = shear_y(img, 0.2, [128]*3)
    plt.imshow(aug_sheary)
    plt.show()
    aug_sheary.save(os.path.join(images_folder, "shear_y.jpg"))


def test_translate():
    img = load_image(input_image)
    aug_translatex = translate_x(img, 50, [128]*3)
    plt.imshow(aug_translatex)
    plt.show()
    aug_translatex.save(os.path.join(images_folder, "translate_x.jpg"))
    aug_translatey = translate_y(img, 50, [128]*3)
    plt.imshow(aug_translatey)
    plt.show()
    aug_translatey.save(os.path.join(images_folder, "translate_y.jpg"))


def test_sharpness():
    img = load_image(input_image)
    aug = sharpness(img, 5)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "sharpness.jpg"))


def test_equalize():
    img = load_image(input_image)
    aug = equalize(img)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "equalize.jpg"))


def test_invert():
    img = load_image(input_image)
    aug = invert(img)
    plt.imshow(aug)
    plt.show()
    aug.save(os.path.join(images_folder, "invert.jpg"))


def test_prepocess_input():
    img = load_image(input_image)
    for i in range(10):
        aug = preprocess_input(img)
        plt.imshow(aug)
        plt.show()
    aug.save(os.path.join(images_folder, "prepocess_input.jpg"))


if __name__ == '__main__':
    # test_cutout()
    #test_contrast()
    # test_autocontrast()
    #test_color()
    # test_solarize()
    # test_solarize_add()
    #test_brightness()
    # test_posterize()
    # test_rotate()
    # test_shear()
    # test_translate()
    #test_sharpness()
    # test_equalize()
    # test_invert()
    # print(_randomly_negate_tensor(1))

    test_prepocess_input()
