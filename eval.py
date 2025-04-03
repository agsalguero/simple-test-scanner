import re
import numpy as np
import argparse
import os

from scanner import printProgressBar

    
if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate answers from CSV files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the student and test answer files.')
    parser.add_argument('--correct', type=float, default=1.0, help='Correct answer score.')
    parser.add_argument('--wrong', type=float, default=-0.25, help='Wrong answer score.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file containing the results of all students.')
    parser.add_argument('--pb', action='store_true', help='Show progress bar.')
    args = parser.parse_args()

    # check if the input file exists
    if not os.path.exists(args.input):
        print(f'Input file {args.input} does not exist.')
        exit(1)

    # get all the files in the input folder witht the format 'student_23.csv', 'student_24.csv', etc.
    students_answers = []
    pattern = re.compile(r'student_(\d+).csv')
    for file in os.listdir(args.input):
        if pattern.match(file):
            student_number = int(pattern.match(file).group(1))
            students_answers.append((student_number, os.path.join(args.input, file)))
    students_answers.sort(key=lambda x: x[0])
    print(f'Found {len(students_answers)} students.')

    # get all the files in the input folder witht the format '_0_answers.csv', '_1_answers.csv', etc.
    test_variants = []
    pattern = re.compile(r'.*_(\d+)_answers.csv')
    for file in os.listdir(args.input):
        if pattern.match(file):
            variant_number = int(pattern.match(file).group(1))
            test_variants.append((variant_number, os.path.join(args.input, file)))
    test_variants.sort(key=lambda x: x[0])

    # load the correct answers for each test variant
    correct_answers = []
    for variant_number, file in test_variants:
        with open(file, 'r') as f:
            correct_answers.append([line.strip().split(',') for line in f.readlines()][0])
    print(f'Found {len(test_variants)} test variants.')

    # create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, 'results.csv'), 'w') as f_results:
        # extract answers from each student's file and compare with the correct answers
        for i, student_answers in students_answers:
            
            if args.pb:
                printProgressBar(i, len(student_answers), prefix='Processing:', suffix='Complete', length=50)

            # read the student's answers
            with open(student_answers, 'r') as f_student_answers:
                # read the first line of the file and split it by commas
                student_data = [line.strip().split(',') for line in f_student_answers.readlines()][0]

                # select the correct test variant for the student
                test_variant = i % len(test_variants)

                # compare the student's answers with the correct answers
                score = 0
                for j, answer in enumerate(correct_answers[test_variant]):
                    if answer == '':
                        continue
                    elif answer == student_data[j]:
                        score += args.correct
                    else:
                        score += args.wrong

                # write the answers and the score to the output file
                f_results.write(','.join(map(str, student_data)))
                f_results.write(f',{score}\n')        

    if args.pb:
        printProgressBar(len(student_answers), len(student_answers), prefix='Processing:', suffix='Complete', length=50)
