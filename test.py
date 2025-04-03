import re
import random
import argparse
import os

def parse_header(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # read text until '% Questions start here' is found
    header = content.split('% Questions start here')[0].strip()

    return header

def parse_footer(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # read text after '% Questions end here'
    footer = content.split('% Questions end here')[1].strip()

    return footer

def parse_questions(file_path, show_questions=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex to extract questions and categories
    question_pattern = re.compile(r'%-\s*(\w+).*?\n\s*\\item\s(.*?)\n\s*\\begin{enumerate}(.*?)\\end{enumerate}\n', re.DOTALL)

    questions_by_category = {}

    for match in question_pattern.findall(content):
        category_header = match[0].strip().split('_')
        category = category_header[0]  # Extract the category name
        cols = 2 # Default number of columns
        if len(category_header) > 1:
            cols = category_header[1]  # Extract the numbre columns (if any)
        
        question = match[1].strip()
        if show_questions:
            print(f"{category}: {question[:50]}")

        options = match[2].split('\\item')[1:]
        
        if category not in questions_by_category:
            questions_by_category[category] = []
        
        questions_by_category[category].append((question, options, cols))

    return questions_by_category

def randomize_questions(questions_by_category):
    randomized_questions = {}

    # concat all questions in each category
    all_questions = []
    for category, questions in questions_by_category.items():
        all_questions.extend(questions)

    # shuffle all questions
    random.shuffle(all_questions)

    return all_questions

def randomize_options(questions):
    randomized_questions = []

    for question, options, cols in questions:
        correct_answer = options[0]
        random.shuffle(options)
        randomized_questions.append((question, options, options.index(correct_answer), cols))

    return randomized_questions

def generate_latex(output_path, questions, header='', footer='', show_answers=False):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(header)
        file.write('\n% Questions start here\n')
        for question, options, correct, cols in questions:
            file.write(f'\\item {question}\n')
            if int(cols) > 1:
                file.write('\\begin{multicols}{'+str(cols)+'}\n')
            file.write('\t\\begin{enumerate}\n')
            for i, option in enumerate(options):
                if show_answers and i == correct:
                    file.write(f'\t\t\\item $\\blacktriangleright$ {option}\n')
                else:
                    file.write(f'\t\t\\item {option}\n')
            file.write('\t\\end{enumerate}\n\n')
            if int(cols) > 1:
                file.write('\\end{multicols}\n')

        file.write(footer)

def generate_answers(output_path, questions):
    with open(output_path, 'w', encoding='utf-8') as file:
        correct_answers = []
        for question, options, correct, cols in questions:
            correct_answers.append(correct)

        file.write(','.join(map(str, correct_answers)))

if __name__ == "__main__":
    # parse from command line    
    parser = argparse.ArgumentParser(description="Randomize questions and options from a LaTeX file.")
    parser.add_argument("--input", required=True, help="Path to the input LaTeX file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--variations", type=int, default=1, help="Number of variations to generate.")
    parser.add_argument("--answers", action="store_true", help="Generate answers file for each variation.")
    parser.add_argument("--show_questions", action="store_true", help="Show the parsed questions in console.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder.")

    args = parser.parse_args()

    # check if the input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        exit(1)

    # create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # set values
    random.seed(args.seed)
    output_file_name = os.path.splitext(args.input)[0]

    # extract the header and footer from the input file
    header = parse_header(args.input)
    footer = parse_footer(args.input)

    # parse the questions from the input file
    questions_by_category = parse_questions(args.input, show_questions=args.show_questions)
    total_questions = sum(len(q) for q in questions_by_category.values())
    print(f"Total questions: {total_questions}")

    # generate specified number of variations
    for i in range(args.variations):
        # randomize the order of the questions
        randomized_questions = randomize_questions(questions_by_category)

        # randomize the order of the options and store the correct answer index
        randomized_questions = randomize_options(randomized_questions)

        # replace (VARIANT) int the header and footer for the current variant number
        header_variant = header.replace('(VARIANT)', str(i))
        footer_variant = footer.replace('(VARIANT)', str(i))

        # add the variation number to the output file name
        output_file_name = os.path.join(args.output, os.path.basename(output_file_name))
        generate_latex(f"{output_file_name}_{i}.tex", randomized_questions, 
                       header=header_variant, footer=footer_variant, show_answers=False)
        
        if args.answers:
            # generate the answers file for the current variant
            generate_latex(f"{output_file_name}_{i}_answers.tex", randomized_questions, 
                           header=header_variant, footer=footer_variant, show_answers=True)

        # store the correct answer index for each question
        generate_answers(f"{output_file_name}_{i}_answers.csv", randomized_questions)
    