import os
from Config              import config
from PIL                 import Image
from utils               import absolute_dir_wrapper


def resize_images(data_type='train'):
    """Resize the images from 'image_dir' and save into 'output_dir'."""
    if data_type not in ('train', 'test', 'validate'):
        raise ValueError("data_type has to be 'train', 'test' or 'validate'.")

    image_dir, output_dir, image_size = absolute_dir_wrapper(config.get('%s_image_dir' % data_type)), \
                                        absolute_dir_wrapper(config.get('%s_resized_image_dir' % data_type)), \
                                        config.get('image_size')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        img_output_dir = os.path.join(output_dir, image)
        if os.path.exists(img_output_dir):
            continue

        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = img.resize(image_size, Image.ANTIALIAS)
                img.save(img_output_dir, img.format)
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized {} images."
                  .format(i + 1, num_images, data_type))
