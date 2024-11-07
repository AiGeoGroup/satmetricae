import os
import pprint

import matplotlib.pyplot as plt
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

import joblib
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm


def display_labels(data_path='/users/sunyong/dataserver/EuroSAT/2750/'):
    # modify to fit your system
    return os.listdir(data_path)


def resize_all(path_to_data, pklname, includes, width=150, height=None):
    """
    path_to_data: path to data
    pklname: path to output file
    width: target width of the image in pixels
    include: set containing str
    """
    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1})remote sensing images'.format(
        int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(path_to_data):
        if subdir == ".DS_Store":
            pass
        else:
            current_path = os.path.join(path_to_data, subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))  #[:,:,::-1]
                    data['label'].append(subdir[:-2])
                    data['filename'].append(file)
                    data['data'].append(im)

    return data


def get_data_set(data_path='/users/sunyong/dataserver/EuroSAT/2750/'):
    class_names = display_labels(data_path)

    pkl_name = 'eurosat'
    width = 64

    data = resize_all(path_to_data=data_path,
                      pklname=pkl_name,
                      width=width,
                      includes=class_names)
    return data


def display_photos(data):
    # use np.unique to get all unique values in the list of labels

    labels = np.unique(data['label'])

    # set up the matplotlib figure and axes, based on the number of labels
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15, 4)
    fig.tight_layout()

    # make a plot for every label (equipment) type. The index method returns the
    # index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)

        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)


def get_train_test_data(data_path='/users/sunyong/dataserver/EuroSAT/2750/'):
    data = get_data_set(data_path)
    X = np.array(data['data'])  # Numpy Array
    y = np.array(data['label'])  # Label

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test  # 训练数据，训练标签，测试数据，测试标签


def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5

    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]

    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'

    xtemp = np.arange(len(unique))

    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)

# 展示训练集和测试集的标签分布情况plot_amount_photos_per_type，plot_bar
def plot_amount_photos_per_type(y_train, y_test):
    plt.suptitle('relative amount of photos per type')
    plot_bar(y_train, loc='left')
    plot_bar(y_test, loc='right')
    plt.legend([
        'train ({0} photos)'.format(len(y_train)),
        'test ({0} photos)'.format(len(y_test))
    ])

# print(display_labels(data_path='/users/sunyong/dataserver/EuroSAT/2750/'))
# X_train, X_test, y_train, y_test = get_train_test_data()
# plot_amount_photos_per_type(y_train, y_test)


import matplotlib.pyplot as plt

def bar_scatter_plot_tutorial():
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]

    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Categorical Plotting')
    plt.show()


def display_satellite_images(n=4):
    # Data manipulation and visualization
    inputs, classes = next(iter(train_loader))  #
    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    for i in range(n):
        for j in range(n):
            image = inputs[i * n + j].numpy().transpose(
                (1, 2, 0))  # 3 * 64 * 64 === PIL.image: 64*64*3
            image = np.clip(
                np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)
            title = class_names[classes[i * n + j]]
            axes[i, j].imshow(image)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')


import matplotlib.pyplot as plt
import math

# 测试TripletAE网络模型图像生成情况
def plot_test_decoder_images(ae_model, test_loader):
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # 0-255 归一 0-1
    ae_model.eval()

    for step, elem in enumerate(tqdm(test_loader, desc="Testing",
                                     leave=False)):
        (anchor_img, _, _), labels = elem
        anchor_img = anchor_img.to(device)
        pred = ae_model(anchor_img)
        # print(pred.shape)
        n = int(math.sqrt(len(labels)))

        fig, axes = plt.subplots(n, n, figsize=(8, 8))

        for i in range(n):
            for j in range(n):
                image = pred[i * n + j].cpu().detach().numpy().transpose(
                    (1, 2, 0))  # 3 * 64 * 64 === PIL.image: 64*64*3
                image = np.clip(
                    np.array(imagenet_std) * image + np.array(imagenet_mean),
                    0, 1)
                # print(image.shape)
                # print(labels[i * n + j])
                title = test_loader.dataset.classes[labels[i * n + j]]
                axes[i, j].imshow(image)
                axes[i, j].set_title(title)
                axes[i, j].axis('off')
        break


def plot_rand_decoder_images(ae_model, test_loader):
    imagenet_mean, imagenet_std = [0.485, 0.456,
                                   0.406], [0.229, 0.224,
                                            0.225]  # 0-255 归一 0-1
    n = 4

    inputs, classes = next(
        iter(test_loader))  # inputs -> (achors, neighbors, distants)
    a_inputs = inputs[0]
    print(a_inputs.shape)

    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    for i in range(n):
        for j in range(n):
            image = a_inputs[i * n + j].cpu().detach().numpy().transpose(
                (1, 2, 0))  # 3 * 64 * 64 === PIL.image: 64*64*3
            image = np.clip(
                np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)
            title = classes[classes[i * n + j]]
            axes[i, j].imshow(image)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')

    # 重建图像展示
    a_inputs_pred = ae_model(a_inputs)
    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    for i in range(n):
        for j in range(n):
            image = a_inputs_pred[i * n + j].cpu().detach().numpy().transpose(
                (1, 2, 0))  # 3 * 64 * 64 === PIL.image: 64*64*3
            image = np.clip(
                np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)
            title = classes[classes[i * n + j]]
            axes[i, j].imshow(image)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
