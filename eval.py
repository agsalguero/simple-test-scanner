# pylint: disable=missing-module-docstring
import re
import argparse
import os

from scanner import print_progress_bar


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate answers from CSV files.")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the folder containing the student and test answer files.",
    )
    parser.add_argument(
        "--correct", type=float, default=1.0, help="Correct answer score."
    )
    parser.add_argument(
        "--wrong", type=float, default=-0.25, help="Wrong answer score."
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
    parser.add_argument("--pb", action="store_true", help="Show progress bar.")
    parser.add_argument("--decimal", type=str, default=".", help="Decimal separator.")
    parser.add_argument("--separator", type=str, default=",", help="CSV separator.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output CSV file containing the results of all students.",
    )
    args = parser.parse_args()

    # check if the input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        exit(1)

    # get all the files in the input folder witht the format 'student_23.csv',
    # 'student_24.csv', etc.
    students_answers = []
    pattern = re.compile(r"student_(\d+).csv")
    for file in os.listdir(args.input):
        if pattern.match(file):
            student_number = int(pattern.match(file).group(1))
            students_answers.append((student_number, os.path.join(args.input, file)))
    students_answers.sort(key=lambda x: x[0])
    print(f"Found {len(students_answers)} students.")

    # get all the files in the input folder witht the format '_0_answers.csv',
    # '_1_answers.csv', etc.
    test_variants = []
    pattern = re.compile(r".*_(\d+)_answers.csv")
    for file in os.listdir(args.input):
        if pattern.match(file):
            variant_number = int(pattern.match(file).group(1))
            test_variants.append((variant_number, os.path.join(args.input, file)))
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
                [line.strip().split(",") for line in f.readlines()][0]
            )
    print(f"Found {len(test_variants)} test variants.")

    backrefs = []
    if args.backrefs:
        # check if the backrefs file exists
        if not os.path.exists(args.backrefs):
            print("No backrefs file found.")
        else:
            # read the backrefs from the CSV file
            with open(args.backrefs, "r", encoding="utf-8") as f_backrefs:
                backref = [line.strip().split(",") for line in f_backrefs.readlines()][
                    0
                ]
            backref = [int(b) for b in backref]
            backrefs.append(backref)

    # create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, "results.csv"), "w", encoding="utf-8") as f_results:
        # extract answers from each student's file and compare with the correct answers
        for i, student_answers in students_answers:

            if args.pb:
                print_progress_bar(
                    i,
                    len(students_answers),
                    prefix="Processing:",
                    suffix="Complete",
                    length=50,
                )

            # read the student's answers
            with open(student_answers, "r", encoding="utf-8") as f_student_answers:
                # read the first line of the file and split it by commas
                student_data = [
                    line.strip().split(",") for line in f_student_answers.readlines()
                ][0]

                # select the correct test variant for the student
                test_variant = student_variants[i]

                # pylint: disable=invalid-name
                # compare the student's answers with the correct answers
                score = 0
                for j, answer in enumerate(correct_answers[test_variant]):
                    if student_data[j] == "":
                        continue
                    elif answer == student_data[j]:
                        score += args.correct
                        student_data[j] = args.correct
                    else:
                        score += args.wrong
                        student_data[j] = args.wrong

                if args.backrefs:
                    # reorder the student's answers based on the backrefs
                    student_data = [student_data[b] for b in backrefs[test_variant]]

                # write the answers and the score to the output file
                formatted_data = [
                    str(item).replace(".", args.decimal) for item in student_data
                ]
                f_results.write(args.separator.join(formatted_data))
                f_results.write(
                    f"{args.separator}{str(score).replace('.', args.decimal)}\n"
                )

    if args.pb:
        print_progress_bar(
            len(students_answers),
            len(students_answers),
            prefix="Processing:",
            suffix="Complete",
            length=50,
        )
