#!./env/bin/python3

#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
# 
# Source: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
            self,
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath
        ):

        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(
            self,
            images_filepath,
            labels_filepath
        ):

        labels = []

        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))

            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))

            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            
            image_data = array("B", file.read()) 

        images = []

        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    

def write_file():
    pass

    
def main() -> None:
    data_loader = MnistDataloader("../raw/train-images.idx3-ubyte", "../raw/train-labels.idx1-ubyte",
                                 "../raw/t10k-images.idx3-ubyte", "../raw/t10k-labels.idx1-ubyte")
    
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    datasets = [
        (x_train, y_train, "../train/data"),
        (x_test, y_test, "../validate/data")
    ]

    for images, labels, filepath in datasets:
        with open(filepath, "wb") as file:
            total_samples = len(images)
            file.write(struct.pack(">I", total_samples))

            for img, label in zip(images, labels):
                flattened_img = [float(pixel) for row in img for pixel in row]

                one_hot_label = [1.0 if i == label else 0.0 for i in range(10)]

                input_len = len(flattened_img)
                file.write(struct.pack(">I", input_len))
                file.write(struct.pack(f">{input_len}f", *flattened_img))

                output_len = len(one_hot_label)
                file.write(struct.pack(">I", output_len))
                file.write(struct.pack(f">{output_len}f", *one_hot_label))

    print("Files successfully packed!")

if __name__ == "__main__":
    main()