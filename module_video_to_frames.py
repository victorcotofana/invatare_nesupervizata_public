import os
import time
import cv2


IMAGE_SHAPE = (227, 227)
FRAMES_TO_JUMP = 10
SKIP_NEXT_FRAME_INDEX = 5
FROM_PATH = 'C:\\Unsupervised Video Learning\\UCF-101'
TO_PATH = 'C:\\Unsupervised Video Learning\\UCF-101-FRAMES'

selected_categories = [
    'Archery',
    'Billiards',
    'Bowling',
    'GolfSwing',
    'HammerThrow',
    'HorseRace',
    'Rowing',
    'Skiing',
    'SoccerPenalty',
    'TennisSwing'
]


def extract_frames_from_video_to_path_old(from_path, to_path):
    # this method extracts all frames from the video, we don't need that
    vidcap = cv2.VideoCapture(from_path)

    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Num of frames for \'', from_path, '\'')
    print(num_frames)

    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, IMAGE_SHAPE)
        cv2.imwrite(to_path + '/frame_%03d.jpg' % count, image)  # save frame as JPG file
        success, image = vidcap.read()
        count += 1
    print('Frames successfully extracted in: ' + to_path)


def extract_frames_from_video_to_path(from_path, to_path):
    # here we're trying to extract only the frames that we need, aka we jump over several frames

    vidcap = cv2.VideoCapture(from_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Total number of frames aprox.: ', num_frames,
          'Jump index: ', FRAMES_TO_JUMP,
          'Total preprocessed frames aprox.: ', int(num_frames / FRAMES_TO_JUMP))

    output_count_frames = 0
    count_frames = 0

    success_one, image_one = vidcap.read()
    success_two, image_two = vidcap.read()
    while success_one and success_two:
        # we already have two images, we only have to process them

        # read the query frame at every jump_index
        if count_frames % FRAMES_TO_JUMP == 0:
            image_one = cv2.resize(image_one, IMAGE_SHAPE)
            # save frame as JPG file
            cv2.imwrite(to_path + '/query_frame_%05d.jpg' % output_count_frames, image_one)

            # the counter should depend on the query frame, that's why there is -1 in the next if
            output_count_frames += 1

            # possibly read the next frame, verify 5th pair will be negative, so no need to save that one
            if (output_count_frames - 1) % SKIP_NEXT_FRAME_INDEX != 0:
                image_two = cv2.resize(image_two, IMAGE_SHAPE)

                cv2.imwrite(to_path + '/next_frame_%05d.jpg' % (output_count_frames - 1), image_two)

        # the second image is moved to the first slot of the window
        success_one, image_one = success_two, image_two

        # and we read only the next frame
        success_two, image_two = vidcap.read()
        count_frames += 1

    print('Frames successfully extracted in: ' + to_path)


def extract_frames_from_videos(initial_path, output_path):
    # sanity check
    if not os.path.isdir(initial_path):
        raise Exception('initial_path not a dir')

    if not os.path.isdir(output_path):
        raise Exception('output_path not a dir')

    for category_dir in os.listdir(initial_path):
        if category_dir not in selected_categories:
            continue

        input_full_path = os.path.join(initial_path, category_dir)
        if not os.path.isdir(input_full_path):
            # move to the next category folder
            continue

        output_full_path = os.path.join(output_path, category_dir)
        if not os.path.isdir(output_full_path):
            # the output dir does not exist, so we create it
            os.mkdir(output_full_path)

        for video_file in os.listdir(input_full_path):
            full_video_file = os.path.join(input_full_path, video_file)
            if not os.path.isfile(full_video_file):
                # there are no more videos to process in current folder
                continue

            name_of_file = video_file.split('.')[0]
            output_full_video_dir = os.path.join(output_full_path, name_of_file)

            # create a folder for each video in which we put the frames
            os.mkdir(output_full_video_dir)
            extract_frames_from_video_to_path(full_video_file, output_full_video_dir)


if __name__ == "__main__":
    start = time.time()
    extract_frames_from_videos(FROM_PATH, TO_PATH)
    end = time.time()
    print('Preprocessing time: ' + str(end - start) + " seconds")
