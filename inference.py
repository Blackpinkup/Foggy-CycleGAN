import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from IPython.display import clear_output

tfds.disable_progress_bar()

from lib.dataset import DatasetInitializer
datasetInit = DatasetInitializer(256, 256)


## Build Generator
from lib.models import ModelsBuilder
OUTPUT_CHANNELS = 3
models_builder = ModelsBuilder()

use_transmission_map = False #@param{type: "boolean"}
use_gauss_filter = False #@param{type: "boolean"}
use_resize_conv = False #@param{type: "boolean"}
generator_clear2fog = models_builder.build_generator(use_transmission_map=use_transmission_map,
                                                     use_gauss_filter=use_gauss_filter,
                                                     use_resize_conv=use_resize_conv)
generator_fog2clear = models_builder.build_generator(use_transmission_map=False)


## Build Discriminator
use_intensity_for_fog_discriminator = False #@param{type: "boolean"}
discriminator_fog = models_builder.build_discriminator(use_intensity=use_intensity_for_fog_discriminator)
discriminator_clear = models_builder.build_discriminator(use_intensity=False)


## Checkpoints
weights_path = '/home/yjy/RScl/GAN/Foggy-CycleGAN/weights/'

from lib.train import Trainer
trainer = Trainer(generator_clear2fog, generator_fog2clear,
                 discriminator_fog, discriminator_clear)

trainer.configure_checkpoint(weights_path = weights_path, load_optimizers=False)

from lib.plot import plot_generators_predictions
from lib.plot import plot_discriminators_predictions


## Testing
from lib.plot import plot_clear2fog_intensity
from lib.plot import get_clear2fog_intensity
from matplotlib import pyplot as plt


if __name__ == '__main__':

    intensity_path = '/home/yjy/RScl/GAN/Foggy-CycleGAN/intensity_result'
    from lib.tools import create_dir
    create_dir(intensity_path)
    file_path = '/home/yjy/RScl/GAN/data/CODaN/data/train/Bicycle/000000000074.jpg'

    image_clear = tf.io.decode_png(tf.io.read_file(file_path), channels=3)
    image_clear, _ = datasetInit.preprocess_image_test(image_clear, 0)
    step = 0.25
    for (ind, i) in enumerate(tf.range(0.25, 1, step)):
        # fig = plot_clear2fog_intensity(generator_clear2fog, image_clear, i)
        # fig.savefig(os.path.join(intensity_path
        #                         , "{:02d}_intensity_{:0.2f}.jpg".format(ind,i)), bbox_inches='tight', pad_inches=0)
        fig = get_clear2fog_intensity(generator_clear2fog, image_clear, i)
        # fig = np.array(fig)
        # fig.savefig(os.path.join(intensity_path
        #                         , "{:02d}_intensity_{:0.2f}.jpg".format(ind,i)), bbox_inches='tight', pad_inches=0)
        final = Image.fromarray((fig * 255).astype(np.uint8))
        final.save(os.path.join(intensity_path
                                , "{:02d}_intensity_{:0.2f}.jpg".format(ind,i)))