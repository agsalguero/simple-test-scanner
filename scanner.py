# pylint: disable=no-member, missing-module-docstring
import argparse
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
):
    """
    Extract answers from the image using ArUco markers.
    """

    transformed_image = corrected_image(image_path, transf_img_width, transf_img_height)
    if transformed_image is not None:
        selected_answers = []

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

                    # draw a horizontal line on top of the cell indicating the brightness percentage
                    brightness_percentage = 1 - (white_pixels / total_pixels)
                    line_length = int(cell_image.shape[1] * brightness_percentage)
                    cv2.line(
                        transformed_image,
                        (cell_top_left[0] + 1, cell_top_left[1] + 1),
                        (cell_top_left[0] + line_length - 1, cell_top_left[1] + 1),
                        (0, 0, 255),
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
                    cell_bottom_right = int(col_x + cell_width), int(
                        row_y + cell_height
                    )

                    cv2.rectangle(
                        transformed_image,
                        cell_top_left,
                        cell_bottom_right,
                        (0, 255, 0),
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

            cv2.rectangle(
                transformed_image, top_left_corner, bottom_right_corner, (255, 0, 0), 1
            )

            # append the answers in the regions to the global answers list
            selected_answers.append(darker_cells_per_row)

        return transformed_image, selected_answers
    else:
        return None


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract answers from scanned image or pdf."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input image or pdf file."
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

    # extract answers from each image
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
        )
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
                    # check if the answers are correct
                    for j, answer in enumerate(answers):
                        answer = answer if answer != "" else None
                        if answer == true_selected_answers[i][j]:
                            true_selected += 1
                        elif answer is None and true_selected_answers[i][j] is not None:
                            false_empty += 1
                            print(
                                f"False empty: student {i}, question {j}, answer {answer}"
                            )
                        elif answer is not None and true_selected_answers[i][j] is None:
                            false_selected += 1
                            print(
                                f"False selected: student {i}, question {j}, answer {answer}"
                            )
                        else:
                            false_option += 1
                            print(
                                f"False option: student {i}, question {j}, answer {answer}"
                            )
                                                        
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

    # remove the temporary folder and all its contents
    if args.input.lower().endswith(".pdf"):
        shutil.rmtree(temp_dir)
