import linecache
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def get_cmap(n, name='hsv'):
    # returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    # the keyword argument name must be a standard matplotlib colormap name

    return plt.cm.get_cmap(name, n)


def visualize_data_points(embedded, kmeans_index_array, file_name):
    # top level container for all the plot elements
    # figsize: figure dimension in inches
    fig = plt.figure(figsize=(8, 8))

    # add an Axes to the figure as part of a subplot arrangement.
    axes = fig.add_subplot(111)

    # we need kmeans_index_array mainly for determining the colors for the points
    no_clusters = len(set(kmeans_index_array)) + 1
    color_map = get_cmap(no_clusters)

    for i, [x, y] in enumerate(embedded):
        # add to the figure; s: the marker size
        axes.scatter(x, y, s=20, c=color_map(kmeans_index_array[i]))

    fig.savefig(file_name, format='jpeg', dpi=600, bbox_inches='tight')
    plt.close()


def visualize_data_frames(embedded, base_path, read_file_path, file_name):
    # details in the previous method
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111)

    # so that the first iteration will initialize everything
    current_folder_path = None
    read_file_index = 0
    query_frame_index = 0
    no_query_frames = 0

    for index, [x, y] in enumerate(embedded):
        # get the right frame

        # when we went through all frames in the folder, change the path
        if query_frame_index == no_query_frames:
            # read the next line
            read_file_index += 1

            # reads exactly one line from a file. Neat right?
            current_folder_path = linecache.getline(read_file_path, read_file_index)

            # create the eligible path
            current_folder_path = current_folder_path.split('.avi')[0].replace('/', '\\')
            current_folder_path = os.path.join(base_path, current_folder_path)
            all_frames_from_path = os.listdir(current_folder_path)

            # reset the parameters
            no_query_frames = len(list(filter(lambda frame_name: 'query' in frame_name, all_frames_from_path)))
            query_frame_index = 0

        # load the current frame
        current_frame_path = os.path.join(current_folder_path, 'query_frame_%05d.jpg' % query_frame_index)
        query_frame_index += 1
        current_frame = OffsetImage(plt.imread(current_frame_path), zoom=0.02)

        # we need the empty points so that the graph will be rendered where they should be
        axes.scatter(x, y, s=0)

        # otherwise this will only be shown inside a (0,1) box
        ab = AnnotationBbox(current_frame, (x, y), frameon=False)
        axes.add_artist(ab)

    fig.savefig(file_name, format='jpeg', dpi=1000, bbox_inches='tight')
    plt.close()
