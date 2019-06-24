import random
import os
import datetime
import traceback
import h5py
import numpy as np
from keras.models import load_model

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input

import module_siamese_alex_net as siamese_net
import module_spreadsheet as logger
import module_drive as drive
import module_visualization as visualization
import module_evaluation as evaluation


# as, initial thought, run into some errors and can only go to 500
# NO_EPOCHS = 750
# to do the rest, we need only 80 epochs
NO_EPOCHS = 80

# based on the advice of Andrew Ng, models with less than 2000 observations should not do mini-batch
MINI_BATCH_SIZE = 1

TRAIN_LIST_VIDEO_PATH = 'selectedTrainList.txt'
TEST_LIST_VIDEO_PATH = 'selectedTestList.txt'
# preprocessed path
BASE_PATH = 'C:\\Unsupervised Video Learning\\UCF-101-FRAMES'

# aka number of categories
KMEANS_NO_CLUSTERS = 10

SAVE_MODEL_ON_ITERATION = 10
SAVE_FILE_NAME_MODEL = 'siamese_net_model.h5'
SAVE_FILE_NAME_PRED = 'np_array_predict.h5'
H5_PRED_DATASET = 'predictions'
SAVE_FILE_MIMETYPE = 'application/h5'

SAVE_FILE_NAME_SCATTER_POINTS = 'scatter_points.jpeg'
SAVE_FILE_NAME_SCATTER_FRAMES = 'scatter_frames.jpeg'
IMAGE_MIMETYPE = 'image/jpeg'

SKIP_NEXT_FRAME_INDEX = 5


def get_random_frame_random_video_path():
    # list of all categories and select one random
    all_categories = os.listdir(BASE_PATH)
    random_category = random.choice(all_categories)

    # create a new path, get all videos from that path and choose one random
    random_category_path = os.path.join(BASE_PATH, random_category)
    all_videos_of_category = os.listdir(random_category_path)
    random_video = random.choice(all_videos_of_category)

    # return the random frame of the random video of random category
    random_video_path = os.path.join(random_category_path, random_video)
    all_frames_of_video = os.listdir(random_video_path)
    random_frame_random_video = random.choice(all_frames_of_video)
    random_frame_path = os.path.join(random_video_path, random_frame_random_video)

    return random_frame_path


def get_tests_ground_truths():
    # return an array of the categories of the test videos
    # it is not enough to just iterate through the folders, as we test the frames not the videos,
    # so we need to iterate the folder and for each query frame add the ground truth
    test_list_file = open(TEST_LIST_VIDEO_PATH, 'r')

    ground_truths = []
    next_video_line = test_list_file.readline()
    while next_video_line:
        category_next_video = next_video_line.split('/')[0]

        # build the next video folder path
        next_folder_line = next_video_line.split('.avi')[0]
        next_folder = next_folder_line.replace('/', '\\')
        next_folder_path = os.path.join(BASE_PATH, next_folder)

        # check how many query frames there are in the folder
        all_frames_in_folder = os.listdir(next_folder_path)
        query_frames_number = sum(1 if 'query' in frame else 0 for frame in all_frames_in_folder)

        # add for each query frame found the category of the video
        for _ in range(query_frames_number):
            ground_truths.append(category_next_video)

        next_video_line = test_list_file.readline()

    # return the position of the offset to the beginning of the file
    test_list_file.seek(0)
    test_list_file.close()
    return ground_truths


def mini_batch_frames_generator():
    # generate only the input x, the labels are empty

    # initial parameters
    no_frames_iterated = 0
    change_video_folder = True
    next_folder_path = None
    input_flag = []
    input_frames_1 = []
    input_frames_2 = []

    with open(TRAIN_LIST_VIDEO_PATH, 'r') as train_list_file:
        while True:
            # create the mini-batches ourselves
            # line in file: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
            while len(input_flag) < MINI_BATCH_SIZE:
                # keep adding to the batch
                if change_video_folder:
                    change_video_folder = False
                    next_video_line = train_list_file.readline()
                    no_frames_iterated = 0
                    if not next_video_line:
                        # there are no more videos to process, yield the batch and go to next iteration
                        # go to the start of the train list file
                        train_list_file.seek(0)
                        break

                    next_folder_line = next_video_line.split('.avi')[0]
                    next_folder = next_folder_line.replace('/', '\\')
                    next_folder_path = os.path.join(BASE_PATH, next_folder)

                query_frame_path = os.path.join(next_folder_path, 'query_frame_%05d.jpg' % no_frames_iterated)

                # when the change for the video folder is needed
                if not os.path.isfile(query_frame_path):
                    # the next frame does not exist, so we emptied the video folder, we need to go to the next one
                    change_video_folder = True
                    no_frames_iterated = 0
                    continue

                query_frame = load_img(query_frame_path)

                # every 5th frame should be in a negative pair, that is should have a random frame associated to it
                if no_frames_iterated % SKIP_NEXT_FRAME_INDEX != 0:
                    # frames from the same video
                    different_video_frame = 0
                    adjacent_frame_path = os.path.join(next_folder_path, 'next_frame_%05d.jpg' % no_frames_iterated)
                    second_query_frame = load_img(adjacent_frame_path)
                else:
                    # frames from different videos
                    different_video_frame = 1
                    random_frame_path = get_random_frame_random_video_path()
                    second_query_frame = load_img(random_frame_path)

                no_frames_iterated += 1

                # keras expects each input of the mini-batch in it's own array
                input_flag.append([different_video_frame])
                input_frames_1.append(img_to_array(query_frame))
                input_frames_2.append(img_to_array(second_query_frame))

            # generate only the input x without the labels
            if len(input_flag) > 0:
                array_yield = [np.array(input_flag), np.array(input_frames_1), np.array(input_frames_2)]
            else:
                array_yield = None

            yield array_yield

            input_flag = []
            input_frames_1 = []
            input_frames_2 = []


def init_and_train_model(start_time):
    # siamese_model = siamese_net.initialize_model()

    # we load the last saved model, and run the remaining epochs
    siamese_model = siamese_net.get_saved_model('siamese_net_model.h5')

    train_list_file = open(TRAIN_LIST_VIDEO_PATH, 'r')
    drive_file_id = None
    empty_labels = np.zeros(MINI_BATCH_SIZE)

    # iterate through every epoch
    for current_no_epoch in range(NO_EPOCHS):
        print('CURRENT NO. EPOCH: ', current_no_epoch)

        # create mini-batches until you can no more
        for mini_batch in mini_batch_frames_generator():
            # train the model only on current mini batch
            # you have to pass empty labels, otherwise it raises error
            if mini_batch is None:
                break
            siamese_model.train_on_batch(mini_batch, empty_labels)

        is_model_saved = False
        # check if we need to save the model or not
        if current_no_epoch % SAVE_MODEL_ON_ITERATION == 0:
            is_model_saved = True
            # save the model as HDF5 file
            siamese_model.save(SAVE_FILE_NAME_MODEL)
            drive_file_id = drive.update_or_create_file(drive_file_id, SAVE_FILE_NAME_MODEL, SAVE_FILE_MIMETYPE)

        logger.write_line(start_time, 'CURRENT NO. EPOCH: ' + str(current_no_epoch),
                          'MODEL SAVED: ' + str(is_model_saved))

    logger.write_line(start_time, 'CHEER UP, TRAINING FINALLY FINISHED', 'YAAAY')
    train_list_file.close()

    return siamese_model


def test_frames_generator():
    # the generator should generate frame after frame when it is called

    change_video_folder = True
    next_folder_path = None
    no_frames_iterated = 0

    with open(TEST_LIST_VIDEO_PATH, 'r') as test_list_file:
        while True:
            if change_video_folder:
                change_video_folder = False
                next_video_line = test_list_file.readline()
                no_frames_iterated = 0
                if not next_video_line:
                    # there are no more videos to process, stop the testing, we're done
                    # go to the start of the test list file, for whatever reason
                    test_list_file.seek(0)
                    break

                next_folder_line = next_video_line.split('.avi')[0]
                next_folder = next_folder_line.replace('/', '\\')
                next_folder_path = os.path.join(BASE_PATH, next_folder)

            query_frame_path = os.path.join(next_folder_path, 'query_frame_%05d.jpg' % no_frames_iterated)

            # when the change for the video folder is needed
            if not os.path.isfile(query_frame_path):
                # the next frame does not exist, so we emptied the video folder, we need to go to the next one
                change_video_folder = True
                no_frames_iterated = 0
                continue

            query_frame = load_img(query_frame_path)

            no_frames_iterated += 1

            # we should yield only the current frame, without the adjacent frame, without the flag, and wihtout labels
            # keras expects each input to the batch to be its own array
            zero_flag = np.array([[0]])

            query_frame = img_to_array(query_frame)
            query_frame = np.expand_dims(query_frame, axis=0)
            query_frame = preprocess_input(query_frame)
            test_frame = np.array(query_frame)

            zero_filled_frame = np.array([np.zeros(shape=(227, 227, 3))])

            yield [np.array(zero_flag), np.array(test_frame), np.array(zero_filled_frame)]


def predict_model(model, start_time):
    # predictions have the shape of (x, 8193) where x is the input number of frames
    predictions = []
    for test_frame in test_frames_generator():
        frame_prediction = model.predict_on_batch(test_frame)
        predictions.append(frame_prediction)

    logger.write_line(start_time, 'AWESOME! TESTS ARE FINISHED', 'HURRAY')

    # extract only the first 4096 feature spaces, the first element is the flag
    predictions_trimmed = []
    for prediction in predictions:
        # predictions is of shape (1, 8192), aka it is 2D, we need to extract the array
        new_pred = prediction[0]
        predictions_trimmed.append(new_pred[1:4097])

    # save the predicitons
    with h5py.File(SAVE_FILE_NAME_PRED, 'w') as hf:
        hf.create_dataset(H5_PRED_DATASET, data=predictions_trimmed)
        drive.update_or_create_file(None, SAVE_FILE_NAME_PRED, SAVE_FILE_MIMETYPE)

    logger.write_line(start_time, 'THE RESULTS OF THE TEST HAVE BEEN SAVED', 'YEAH')

    return predictions_trimmed


def read_from_file(h5_dataset):
    # how to read the saved h5 file
    with h5py.File(SAVE_FILE_NAME_PRED, 'r') as hf:
        data = hf[h5_dataset][:]

    return data


def evaluate_results(predictions, start_time):
    # Kmeans cluster indices for each frame
    predictions_clusterized = evaluation.apply_kmeans_clustering(predictions, KMEANS_NO_CLUSTERS)

    # conditional entropy on the clustered space
    ground_truths = get_tests_ground_truths()
    conditional_entropy = evaluation.calc_conditional_entropy(predictions_clusterized,
                                                              KMEANS_NO_CLUSTERS,
                                                              ground_truths)
    # tSNE: from 4096d to 2d (x, y)
    predictions_embedded = evaluation.generate_2d_embedding(predictions)

    # generate the matplotlib visualization of the results and save them in drive
    visualization.visualize_data_points(predictions_embedded, predictions_clusterized,
                                        SAVE_FILE_NAME_SCATTER_POINTS)
    drive.update_or_create_file(None, SAVE_FILE_NAME_SCATTER_POINTS, IMAGE_MIMETYPE)

    # visualization.visualize_data_frames(predictions_embedded, BASE_PATH,
    #                                     TEST_LIST_VIDEO_PATH, SAVE_FILE_NAME_SCATTER_FRAMES)
    # drive.update_or_create_file(None, SAVE_FILE_NAME_SCATTER_FRAMES, IMAGE_MIMETYPE)

    logger.write_line(start_time, 'CONGRATULATIONS WE\'RE DONE', 'CONDITIONAL ENTROPY: ' + str(conditional_entropy))

    return conditional_entropy


if __name__ == '__main__':
    module_start_time = datetime.datetime.now()
    try:
        # module_trained_model = init_and_train_model(module_start_time)

        # the training of the model is done, now we want to redo the prediction part
        module_trained_model = siamese_net.get_saved_model('siamese_net_model.h5')

        module_predictions = predict_model(module_trained_model, module_start_time)

        module_conditional_entropy = evaluate_results(module_predictions, module_start_time)

        print('CONDITIONAL ENTROPY: ', module_conditional_entropy)
        print('TIME: ', logger.get_hours_passed(module_start_time))
    except:
        # hope, really really hope, this will never run
        logger.write_line(module_start_time, 'ERROR', traceback.format_exc())
        print(traceback.format_exc())
