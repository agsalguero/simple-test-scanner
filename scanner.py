# pylint: disable=no-member, missing-module-docstring
import argparse
import re
import tempfile
import os
import platform
import shutil
import cv2
import numpy as np
import PIL.Image
from pdf2image import convert_from_path


def print_progress_bar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    print_end="\r",
):
    """
    Call in a loop to create terminal progress bar.
    To enable in Pycharm: edit project configuration -> Execution -> Enable "Emulate terminal 
    in output console"
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar_symbol = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar_symbol}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def find_poppler_path():
    """
    Find the path to the poppler library.
    """
    if platform.system() == "Windows":
        # check under subdirectories of the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for root, _, files in os.walk(current_dir):
            if "pdftoppm.exe" in files:
                return root

        # if not found, check if poppler is in the PATH environment variable
        for path in os.environ["PATH"].split(os.pathsep):
            if os.path.exists(os.path.join(path, "pdftoppm.exe")):
                return path

        # if not found, return None
        return None
    else:
        # for Linux and MacOS, return None as it is usually installed in /usr/local/bin or /usr/bin
        return None


def convert_pdf_to_jpg(pdf, jpg, poppler):
    """
    Convert pdf file to jpg file.
    """

    PIL.Image.MAX_IMAGE_PIXELS = 1418505068
    pages = convert_from_path(
        pdf,
        dpi=300,
        fmt="jpg",
        jpegopt={"quality": 95},  # first_page=1, last_page=1,
        thread_count=8,
        size=(None, 1080),
        poppler_path=poppler,
    )

    images = []
    if len(pages) > 0:
        # Save each page as a jpg file
        for p, page in enumerate(pages):
            student_path = jpg.replace("${i}", str(p))

            # Convert to grayscale
            page = page.convert("L")

            # Apply some gaussian blur to the image
            # page = page.filter(PIL.ImageFilter.GaussianBlur(radius=1))

            # Quantize the image to 4 colors
            # page = page.convert('L').convert('P', palette=PIL.Image.ADAPTIVE, colors=4)

            # Binarize the image
            # page = page.convert('L')
            # threshold = 128
            # page = page.point(lambda p: 255 if p > threshold else 0, '1')

            page.save(student_path, "JPEG")
            images.append(student_path)
    else:
        print(f"Error converting {pdf} to jpg. Pages length is 0.")

    return images


def corrected_image(image_path, transf_img_width=500, transf_img_height=700):
    """
    Correct the perspective of the image using ArUco markers.
    """

    # Load the image
    image = cv2.imread(image_path)

    # Define the ArUco dictionary with the specified parameters (4x4, 250)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Create ArUco parameters for detection
    aruco_parameters = cv2.aruco.DetectorParameters()

    # pylint: disable=unpacking-non-sequence
    # Detect ArUco markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(
        image, aruco_dictionary, parameters=aruco_parameters
    )

    transformed_image = None

    # Check if 4 markers were detected
    if len(corners) == 4:
        # print the ids of the detected markers
        # print(ids)

        # Find index of the marker with id 0, which is the top-left marker
        top_left_marker_index = np.where(ids == 0)[0][0]
        top_right_marker_index = np.where(ids == 1)[0][0]
        bottom_left_marker_index = np.where(ids == 2)[0][0]
        bottom_right_marker_index = np.where(ids == 3)[0][0]

        # Get the corners of each marker
        top_left_marker_corners = corners[top_left_marker_index][0].reshape(
            (4, 2)
        )  # 4 corners, 2 coordinates per corner (x, y)
        top_right_marker_corners = corners[top_right_marker_index][0].reshape((4, 2))
        bottom_left_marker_corners = corners[bottom_left_marker_index][0].reshape(
            (4, 2)
        )
        bottom_right_marker_corners = corners[bottom_right_marker_index][0].reshape(
            (4, 2)
        )

        # Sort the corners of each marker to top-left, top-right, bottom-right, bottom-left
        top_left_marker_sorted_corners = [
            min(top_left_marker_corners, key=lambda x: x[0] + x[1]),
            max(top_left_marker_corners, key=lambda x: x[0] - x[1]),
            max(top_left_marker_corners, key=lambda x: x[0] + x[1]),
            min(top_left_marker_corners, key=lambda x: x[0] - x[1]),
        ]

        top_right_marker_sorted_corners = [
            min(top_right_marker_corners, key=lambda x: x[0] + x[1]),
            max(top_right_marker_corners, key=lambda x: x[0] - x[1]),
            max(top_right_marker_corners, key=lambda x: x[0] + x[1]),
            min(top_right_marker_corners, key=lambda x: x[0] - x[1]),
        ]

        bottom_left_marker_sorted_corners = [
            min(bottom_left_marker_corners, key=lambda x: x[0] + x[1]),
            max(bottom_left_marker_corners, key=lambda x: x[0] - x[1]),
            max(bottom_left_marker_corners, key=lambda x: x[0] + x[1]),
            min(bottom_left_marker_corners, key=lambda x: x[0] - x[1]),
        ]

        bottom_right_marker_sorted_corners = [
            min(bottom_right_marker_corners, key=lambda x: x[0] + x[1]),
            max(bottom_right_marker_corners, key=lambda x: x[0] - x[1]),
            max(bottom_right_marker_corners, key=lambda x: x[0] + x[1]),
            min(bottom_right_marker_corners, key=lambda x: x[0] - x[1]),
        ]

        sorted_corners = np.array(
            [
                top_left_marker_sorted_corners[0],
                top_right_marker_sorted_corners[1],
                bottom_right_marker_sorted_corners[2],
                bottom_left_marker_sorted_corners[3],
            ],
            dtype="float32",
        )

        # Define the size of the new image
        destination_corners = np.array(
            [
                [0, 0],
                [transf_img_width - 1, 0],
                [transf_img_width - 1, transf_img_height - 1],
                [0, transf_img_height - 1],
            ],
            dtype="float32",
        )

        # Get the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(sorted_corners, destination_corners)

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(
            image, matrix, (transf_img_width, transf_img_height)
        )

    return transformed_image


def extract_answers(
    image_path,
    answer_regions,
    cols=4,
    transf_img_width=500,
    transf_img_height=700,
    thresshold=8,
    show_regions=False,
    known_answers=None,
):
    """
    Extract answers from the image using ArUco markers.
    """

    transformed_image = corrected_image(image_path, transf_img_width, transf_img_height)
    if transformed_image is not None:
        selected_answers = []

        answers_in_previous_regions = 0
        # process each region
        for ri, (top_left_corner, bottom_right_corner, rows_in_region) in enumerate(answer_regions):

            # convert percentage to pixel coordinates
            top_left_corner = (
                int(top_left_corner[0] * transf_img_width),
                int(top_left_corner[1] * transf_img_height),
            )
            bottom_right_corner = (
                int(bottom_right_corner[0] * transf_img_width),
                int(bottom_right_corner[1] * transf_img_height),
            )

            # cv2.rectangle(transformed_image, answer_matrix_top_left_corner,
            # answer_matrix_bottom_right_corner, (255, 0, 0), 1)

            cell_width = (bottom_right_corner[0] - top_left_corner[0]) / cols
            cell_height = (bottom_right_corner[1] - top_left_corner[1]) / rows_in_region

            # find the brightness of the pixels in the answer matrix for each cell
            brightness_matrix = np.zeros((rows_in_region, cols))
            for ri in range(rows_in_region):
                row_y = int(top_left_corner[1] + (ri * cell_height))

                for j in range(cols):
                    col_x = int(top_left_corner[0] + (j * cell_width))

                    cell_top_left = col_x, row_y
                    cell_bottom_right = int(col_x + cell_width), int(
                        row_y + cell_height
                    )

                    # get subset of the image for the cell
                    cell_image = transformed_image[
                        cell_top_left[1] : cell_bottom_right[1],
                        cell_top_left[0] : cell_bottom_right[0],
                    ]
                    # cv2.imshow('cell image', cell_image)
                    # cv2.waitKey(0)

                    # draw a circle in the center of the cell
                    center_x = int(cell_top_left[0] + cell_width / 2)
                    center_y = int(cell_top_left[1] + cell_height / 2)
                    radius = int(min(cell_width, cell_height) / 2)
                    if show_regions:
                        cv2.circle(transformed_image, (center_x, center_y), radius, (255, 127, 127), 1)
                        # cv2.imshow('cell image', cell_image)
                        # cv2.waitKey(0)

                    # convert the image to grayscale
                    cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow('cell image', cell_image)
                    # cv2.waitKey(0)

                    # maximize the contrast of the image
                    cell_image = cv2.normalize(
                        cell_image, None, 0, 255, cv2.NORM_MINMAX
                    )
                    # cv2.imshow('cell image', cell_image)
                    # cv2.waitKey(0)

                    # binarize the image
                    # _, cell_image = cv2.threshold(
                    #     cell_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    # )
                    # cv2.imshow('cell image', cell_image)
                    # cv2.waitKey(0)

                    # get the mean brightness of the cell
                    # mean_brightness = np.mean(cell_image)
                    # min_brightness = min(min_brightness, mean_brightness)
                    # max_brightness = max(max_brightness, mean_brightness)

                    # count the number of black pixels in the cell
                    # white_pixels = np.sum(cell_image >= 127)

                    # count the number of of white pixels int circle
                    white_pixels = 0
                    total_pixels = 0
                    # distance_attenuation_factor = 0.5
                    for x in range(cell_image.shape[0]):
                        for y in range(cell_image.shape[1]):
                            # check if the pixel is inside the circle
                            if (x - cell_image.shape[0] / 2) ** 2 + (
                                y - cell_image.shape[1] / 2
                            ) ** 2 <= radius**2:
                                
                                # calculate the relative distance from the center of the circle
                                distance = ((x - cell_image.shape[0] / 2) ** 2 + (y - cell_image.shape[1] / 2) ** 2) ** 0.5
                                r_distance = distance / radius

                                # the further the pixel is from the center, the less it counts                                                               
                                total_pixels += 1 * (1 - r_distance)
                                if cell_image[x, y] >= 230:
                                    white_pixels += 1 * (1 - r_distance)

                    # brightness_matrix[i, j] = mean_brightness
                    brightness_matrix[ri, j] = white_pixels

                    if show_regions:                        
                        # draw a horizontal line on top of the cell indicating the brightness percentage
                        brightness_percentage = 1 - (white_pixels / total_pixels)
                        line_length = int(cell_image.shape[1] * brightness_percentage)
                        cv2.line(
                            transformed_image,
                            (cell_top_left[0] + 1, cell_top_left[1] + 1),
                            (cell_top_left[0] + line_length - 1, cell_top_left[1] + 1),
                            (0, 0, 0),
                            1,
                        )

            # print the brightness matrix with two decimal places
            # np.set_printoptions(precision=2)
            # print(brightness_matrix)

            # get normalized brightness of the cells in the row
            # brightness_matrix_normalized = np.zeros((rows, cols))
            # for i in range(rows):
            #     for j in range(cols):
            #         brightness_matrix_normalized[i, j] = (brightness_matrix[i, j] -
            #  min_brightness) / (max_brightness - min_brightness)

            # print the normalized brightness matrix with two decimal places without
            # using scientific
            # notation
            # np.set_printoptions(precision=2, suppress=True)
            # print(brightness_matrix_normalized)

            # save the indices of the darker cells in each row, None if not dark enough
            darker_cells_per_row = []
            for ri in range(rows_in_region):
                if known_answers:
                    answer_index = answers_in_previous_regions + ri
                    if known_answers[answer_index] is None:
                        darker_cells_per_row.append(None)
                        std_dev = 0
                    else:
                        darker_cells_per_row.append(known_answers[answer_index])
                        std_dev = thresshold + 1
                else:
                    darker_cells_per_row.append(np.argmin(brightness_matrix[ri]))

                    # calc standard deviation of the row
                    std_dev = np.std(brightness_matrix[ri])

                # calc max and min of the other cells in the row
                # max_brightness_others = np.max(np.delete(brightness_matrix[i],
                # darker_cells_per_row[i]))
                # min_brightness_others = np.min(np.delete(brightness_matrix[i],
                # darker_cells_per_row[i]))

                # check if the difference of the brightness of the darker cell
                # and the other cells is
                # greater than ten percent
                # others_diff = max_brightness_others - min_brightness_others
                # darker_diff = brightness_matrix[i, darker_cells_per_row[i]] -
                # min_brightness_others
                # rel_diff = (darker_diff / others_diff)

                # print(brightness_matrix[i], std_dev)

                if (
                    std_dev > thresshold
                ):  # rel_diff < -1 and std_dev > 0.012 and
                    # brightness_matrix[i, int(darker_cells_per_row[i])] < 0.94:
                    # draw a green rectangle around the darker cell
                    row_y = int(top_left_corner[1] + (ri * cell_height))
                    col_x = int(
                        top_left_corner[0] + (darker_cells_per_row[ri] * cell_width)
                    )

                    cell_top_left = col_x, row_y
                    cell_bottom_right = int(col_x + cell_width - 1), int(
                        row_y + cell_height - 1
                    )

                    cv2.rectangle(
                        transformed_image,
                        cell_top_left,
                        cell_bottom_right,
                        (0, 0, 255),
                        1,
                    )

                else:
                    # set the index to None if the cell is not dark enough
                    darker_cells_per_row[ri] = None

            # for i in range(rows):
            #     row_y = int(answer_matrix_top_left_corner[1] + (i * cell_height))
            #     for j in range(cols):
            #         col_x = int(answer_matrix_top_left_corner[0] + (j * cell_width))

            #         cell_top_left = col_x, row_y
            #         cell_bottom_right = int(col_x + cell_width), int(row_y + cell_height)

            #         # calculate the opacity based on the mean brightness
            #         strength = int(brightness_matrix[i, j] * 255)
            #         #print(strength)

            #         # draw the rectangle
            #         cv2.rectangle(transformed_image, cell_top_left,
            # cell_bottom_right, (0, 255-strength, 0), 2)

            if show_regions:                
                cv2.rectangle(
                    transformed_image, top_left_corner, bottom_right_corner, (255, 0, 0), 1
                )

            # append the answers in the regions to the global answers list
            selected_answers.append(darker_cells_per_row)

            answers_in_previous_regions += rows_in_region

        return transformed_image, selected_answers
    else:
        return None
    

def mark_correct_answers(
    image_path,
    answer_regions,
    corrects,
    cols=4,
    transf_img_width=500,
    transf_img_height=700,
):
    """
    Mark the correct answers in the image
    """

    # load image
    image = cv2.imread(image_path)

    next_answer_index = 0
    # process each region
    for ri, (top_left_corner, bottom_right_corner, rows_in_region) in enumerate(
        answer_regions
    ):
        # convert percentage to pixel coordinates
        top_left_corner = (
            int(top_left_corner[0] * transf_img_width),
            int(top_left_corner[1] * transf_img_height),
        )
        bottom_right_corner = (
            int(bottom_right_corner[0] * transf_img_width),
            int(bottom_right_corner[1] * transf_img_height),
        )

        cell_width = (bottom_right_corner[0] - top_left_corner[0]) / cols
        cell_height = (bottom_right_corner[1] - top_left_corner[1]) / rows_in_region

        answers_in_region = corrects[next_answer_index : next_answer_index + rows_in_region]
        # draw a rectangle around the correct answers
        for row in range(len(answers_in_region)):
            col_x = int(
                top_left_corner[0] + (corrects[next_answer_index + row] * cell_width)
            )
            row_y = int(top_left_corner[1] + (row * cell_height))

            cell_top_left = col_x, row_y
            cell_bottom_right = int(col_x + cell_width - 1), int(row_y + cell_height - 1)

            cv2.rectangle(
                image, cell_top_left, cell_bottom_right, (0, 255, 0), 1
            )
        next_answer_index += rows_in_region

    # save the image with the correct answers marked
    cv2.imwrite(image_path, image)


def print_original_questions_number(
    image_path,
    answer_regions,
    original_numbers,
    transf_img_width=500,
    transf_img_height=700,
):
    """
    Print the original question number in the image after last answer in the row
    """
    # load image
    image = cv2.imread(image_path)

    next_answer_index = 0
    # process each region
    for ri, (top_left_corner, bottom_right_corner, rows_in_region) in enumerate(
        answer_regions
    ):
        # convert percentage to pixel coordinates
        top_left_corner = (
            int(top_left_corner[0] * transf_img_width),
            int(top_left_corner[1] * transf_img_height),
        )
        bottom_right_corner = (
            int(bottom_right_corner[0] * transf_img_width),
            int(bottom_right_corner[1] * transf_img_height),
        )

        cell_height = (bottom_right_corner[1] - top_left_corner[1]) / rows_in_region

        answers_in_region = original_numbers[next_answer_index : next_answer_index + rows_in_region]
        # draw a rectangle around the correct answers
        for row in range(len(answers_in_region)):
            col_x = int(
                bottom_right_corner[0]
            )
            row_y = int(top_left_corner[1] + ((row+1) * cell_height))

            cell_top_left = col_x, row_y

            cv2.putText(
                image, str(answers_in_region[row]), cell_top_left, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (127, 127, 127), 1, cv2.LINE_AA
            )
        next_answer_index += rows_in_region

    # save the image with the correct answers marked
    cv2.imwrite(image_path, image)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract answers from scanned image or pdf."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input image or pdf file."
    )
    parser.add_argument("--students-answers", 
        type=str, help="Path to csv file with the answers of the students. " \
        "If not provided, the answers will be extracted from the images."
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        action="append",
        help="Region dimensions (%) in the format top_left_x,top_left_y,bottom_right_x," \
        "bottom_right_y,rows. Can be repeated for multiple regions.",
    )
    parser.add_argument(
        "--thresshold",
        type=float,
        default=6,
        help="Threshold for the standard deviation of the brightness of the cells.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output folder."
    )
    parser.add_argument("--pb", action="store_true", help="Show progress bar.")
    parser.add_argument(
        "--check",
        type=str,
        required=False,
        help="Check the results of the extraction against the known answers in the " \
        "specified csv file.",
    )
    parser.add_argument(
        "--correct", type=float, default=1.0, help="Correct answer score."
    )
    parser.add_argument(
        "--wrong", type=float, default=-0.25, help="Wrong answer score."
    )
    parser.add_argument(
        "--answers",
        type=str,
        required=True,
        help="Path to the folder containing the answer files of the test variants " \
        "(test_i_answers.tex).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        required=False,
        help="Path to the CSV file containing the variant for each student.",
    )
    parser.add_argument(
        "--backrefs",
        type=str,
        required=False,
        help="Path to the CSV file containing the questions backrefs to the " \
        "original questions for each test variant.",
    )
    parser.add_argument(
        "--show-regions",
        action="store_true",
        help="Show the regions in the image.",
    )
    parser.add_argument("--decimal", type=str, default=".", help="Decimal separator.")
    parser.add_argument("--separator", type=str, default=",", help="CSV separator.")

    args = parser.parse_args()

    # check if the input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        exit(1)

    # find poppler
    poppler_path = find_poppler_path()
    if poppler_path is None:
        print(
            "Poppler not found. Please install Poppler and add it to your PATH or " \
            "copy it to the current directory."
        )
        exit(1)

    # top left coordinates of top left marker are (0, 0) and bottom
    # right coordinates of bottom right marker are (1, 1)
    # regions = [
    #     ((0.100, 0.275), (0.24, 0.965), 20), # top left, bottom right, rows
    #     ((0.335, 0.275), (0.475, 0.965), 20),
    #     ((0.57, 0.275), (0.711, 0.965), 20),
    #     ((0.805, 0.275), (0.944, 0.965), 20)
    # ]
    regions = []
    for region in args.region:
        try:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y, rows = map(
                float, region.split(",")
            )
            regions.append(
                ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y), int(rows))
            )
        except ValueError:
            print(
                f"Invalid region format: {region}. Expected format: top_left_x,top_left_y,"\
                "bottom_right_x,bottom_right_y,rows."
            )
            continue

    input_images = []
    # if the input is a pdf file, create a temporary folder and convert all pages to jpg
    # otherwise, use the input image directly
    if args.input.lower().endswith(".pdf"):
        temp_dir = tempfile.mkdtemp()
        pdf_path = args.input
        jpg_path = os.path.join(temp_dir, "student_${i}.jpg")

        input_images = convert_pdf_to_jpg(pdf_path, jpg_path, poppler_path)
    else:
        input_images = [args.input]

    # create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # some statistics about the image processing
    true_selected = 0  # the option selected is correct
    false_option = 0  # incorrect option selected
    false_empty = 0  # none option is selected but someone should be
    false_selected = 0 # some option is selected but should not be

    if args.check:
        # check if the check file exists
        if not os.path.exists(args.check):
            print(f"Check file {args.check} does not exist.")
            exit(1)

        # read the answers from the csv file
        with open(args.check, "r", encoding="utf-8") as f_check:
            true_selected_answers = []
            for line in f_check.readlines():
                # read the first line and split by comma
                true_selected_answers.append(line.strip().split(","))

        # conver empty strings to None
        for i, answers in enumerate(true_selected_answers):
            true_selected_answers[i] = [
                int(a) if a != "" else None for a in answers
            ]

    students_answers = []
    if args.students_answers:
        # check if the students answers file exists
        if not os.path.exists(args.students_answers):
            print(f"Students answers file {args.students_answers} does not exist.")
            exit(1)

        # read the answers from the csv file
        with open(args.students_answers, "r", encoding="utf-8") as f_students_answers:
            for line in f_students_answers.readlines():
                # read the first line and split by comma
                if len(line.strip()) > 0:
                    answers_items = line.strip().split(",")
                    # convert the answers to integers and None
                    answers = [
                        int(a) if a != "" else None for a in answers_items
                    ]
                    students_answers.append(answers)

    # extract answers from each image
    check_exceeded = False
    for i, input_image in enumerate(input_images):

        if args.pb:
            print_progress_bar(
                i, len(input_images), prefix="Processing:", suffix="Complete", length=50
            )

        transformed_img, answers = extract_answers(
            input_image,
            regions,
            transf_img_height=1080,
            transf_img_width=800,
            thresshold=args.thresshold,
            show_regions=args.show_regions,
            known_answers=students_answers[i] if args.students_answers else None,
        )

        if not args.students_answers:
            students_answers.append(answers)

        if transformed_img is not None:
            # cv2.imshow('Answers', transformed_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # save the transformed image to the output folder
            output_image_path = os.path.join(args.output, f"student_{i}.jpg")
            cv2.imwrite(output_image_path, transformed_img)

            # save the answers to a text file
            output_text_path = os.path.join(args.output, f"student_{i}.csv")
            with open(output_text_path, "w", encoding="utf-8") as f:
                # flatten the list of answers and remove None values
                answers = [
                    item if item is not None else ""
                    for sublist in answers
                    for item in sublist
                ]
                f.write(",".join(map(str, answers)))

                # update the statistics
                if args.check:
                    # compare the answers with the true selected answers
                    for j, answer in enumerate(answers):
                        if j < len(true_selected_answers):
                            answer = answer if answer != "" else None
                            if answer == true_selected_answers[j]:
                                true_selected += 1
                            elif answer is None and true_selected_answers[j] is not None:
                                false_empty += 1
                                # print(
                                #     f"False empty: student {i}, question {j}, answer {answer}"
                                # )
                            elif answer is not None and true_selected_answers[j] is None:
                                false_selected += 1
                                # print(
                                #     f"False selected: student {i}, question {j}, answer {answer}"
                                # )
                            else:
                                false_option += 1
                                # print(
                                #     f"False option: student {i}, question {j}, answer {answer}"
                                # )
                        else:
                            check_exceeded = True
                                                        
        else:
            print("No ArUco markers detected in the image")

    if args.pb:
        print_progress_bar(
            len(input_images),
            len(input_images),
            prefix="Processing:",
            suffix="Complete",
            length=50,
        )

    # print the statistics
    if args.check:
        print(f"True detected: {true_selected}")
        print(f"False option: {false_option}")
        print(f"False empty: {false_empty}")
        print(f"False selected: {false_selected}")
        if check_exceeded:
            print(
                f"Warning: Only the first {len(true_selected_answers)} answers were checked."
            )

    # get all the files in the input folder witht the format '_0_answers.csv',
    # '_1_answers.csv', etc.
    test_variants = []
    pattern = re.compile(r".*_(\d+)_answers.csv")
    for file in os.listdir(args.answers):
        if pattern.match(file):
            variant_number = int(pattern.match(file).group(1))
            test_variants.append((variant_number, os.path.join(args.answers, file)))
    test_variants.sort(key=lambda x: x[0])

    student_variants = []
    if args.variants:
        # check if the variants file exists
        if not os.path.exists(args.variants):
            print(f"Variants file {args.variants} does not exist.")
            exit(1)

        # read the variants from the CSV file
        with open(args.variants, "r", encoding="utf-8") as f_variants:
            student_variants = [
                line.strip().split(",") for line in f_variants.readlines()
            ][0]
        student_variants = [int(v) for v in student_variants]

        # check if the number of students and variants match
        if len(student_variants) != len(students_answers):
            print("Number of students and variants do not match.")
            exit(1)
    else:
        # assign the test variant based on the student number
        student_variants = [
            i % len(test_variants) for i in range(len(students_answers))
        ]

    # load the correct answers for each test variant
    correct_answers = []
    for variant_number, file in test_variants:
        with open(file, "r", encoding="utf-8") as f:
            correct_answers.append(
                [int(x) for x in [line.strip().split(",") for line in f.readlines()][0]]
            )
    print(f"Found {len(test_variants)} test variants.")

    # for each student, mark the correct answers in the image
    for i, student_answers in enumerate(students_answers):
        mark_correct_answers(
            os.path.join(args.output, f"student_{i}.jpg"),
            regions,
            correct_answers[student_variants[i]],
            transf_img_height=1080,
            transf_img_width=800,
        )

    # @todo make the answers optional if backrefs are provided?
    backrefs = []
    if args.backrefs:
        # check if the backrefs file exists
        if not os.path.exists(args.backrefs):
            print("No backrefs file found.")
        else:
            # read the backrefs from the CSV file
            with open(args.backrefs, "r", encoding="utf-8") as f_backrefs:
                backref = [line.strip().split(",") for line in f_backrefs.readlines()]
                for br in backref:
                    backrefs.append([int(b) for b in br])

            for i, student_answers in enumerate(students_answers):
                reordered_answers = [q+1 for q in backrefs[student_variants[i]]]
                print_original_questions_number(
                    os.path.join(args.output, f"student_{i}.jpg"),
                    regions,
                    reordered_answers,
                    transf_img_height=1080,
                    transf_img_width=800,
                )

    if not args.students_answers:
        # flatten the list of answers
        students_answers = [
            item for sublist in students_answers for item in sublist
        ]

    with open(os.path.join(args.output, "answers.csv"), "w", encoding="utf-8") as f_answers:
        for i, student_answers in enumerate(students_answers):
            # write the answers to the output file
            formatted_data = [
                str(item).replace(".", args.decimal) if item is not None else "" for item in student_answers
            ]
            f_answers.write(args.separator.join(formatted_data) + "\n")

    with open(os.path.join(args.output, "results.csv"), "w", encoding="utf-8") as f_results:
        # extract answers from each student's file and compare with the correct answers
        for i, student_answers in enumerate(students_answers):
            # select the correct test variant for the student
            test_variant = student_variants[i]

            # pylint: disable=invalid-name
            # compare the student's answers with the correct answers
            score = 0
            for j, answer in enumerate(correct_answers[test_variant]):
                if student_answers[j] is None:
                    continue
                if answer == student_answers[j]:
                    score += args.correct
                    student_answers[j] = args.correct
                else:
                    score += args.wrong
                    student_answers[j] = args.wrong

            if args.backrefs:
                # reorder the student's answers based on the backrefs
                # br = backrefs[test_variant]
                # reordered_answers = [student_answers[i][b] for b in br]
                # student_answers[i] = reordered_answers
                student_answers = [student_answers[b] for b in backrefs[test_variant]]

            # write the answers and the score to the output file
            formatted_data = [
                str(item).replace(".", args.decimal) if item is not None else "" for item in student_answers
            ]
            f_results.write(args.separator.join(formatted_data))
            f_results.write(args.separator + str(test_variant))
            f_results.write(
                f"{args.separator}{str(score).replace('.', args.decimal)}\n"
            )

    # remove the temporary folder and all its contents
    if args.input.lower().endswith(".pdf"):
        shutil.rmtree(temp_dir)
