import tensorflow as tf
import cv2
import numpy as np
import h5py
from glob import glob

FLAGS = tf.compat.v1.app.flags.FLAGS

def degrade(image):
    """Load image at `input_path`, distort and save as `output_path`"""
    SHIFT = 2
    to_swap = np.random.choice([False, True], image.shape[:2], p=[.8, .2])
    swap_indices = np.where(to_swap[:-SHIFT] & ~to_swap[SHIFT:])
    swap_vals = image[swap_indices[0] + SHIFT, swap_indices[1]]
    image[swap_indices[0] + SHIFT, swap_indices[1]] = image[swap_indices]
    image[swap_indices] = swap_vals
    return image

def main(_):
    dirs = glob(FLAGS.data_dir + "/*")
    # file_ = h5py.File(FLAGS.output_file, "w")
    input_data = []
    output_data = []
    for dir_ in dirs:
        images = glob(dir_ + "/*")
        img_counts = 0
        for image in images:
            if img_counts > FLAGS.img_counts_thresh:
                break
            img = cv2.imread(image)
            img = cv2.resize(img, (240, 240))
            output_image = degrade(img.copy())
            input_data.append(img.astype(np.float32) / 255)
            output_data.append(output_image.astype(np.float32) / 255)
            img_counts += 1
    np.save("input_data_" + FLAGS.output_file, np.array(input_data, dtype=np.float32))
    np.save("output_data_" + FLAGS.output_file, np.array(output_data, dtype=np.float32))
    # file_.create_dataset("input_data", data=np.array(input_data, dtype=np.float32))
    # file_.create_dataset("output_data", data=np.array(output_data, dtype=np.float32))
    # file_.close()


if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_string("data_dir", "data directory", "data directory")
    tf.compat.v1.app.flags.DEFINE_string("output_file", "output file", "output file")
    tf.compat.v1.app.flags.DEFINE_integer("img_counts_thresh", 40, "output file")
    tf.compat.v1.app.run()