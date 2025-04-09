# pylint: disable=missing-module-docstring
import re
import random
import argparse
import os


def parse_header(file_path):
    """Parse the header of the LaTeX file to extract the content before the questions start."""
    with open(file_path, "r", encoding="utf-8") as tex_file:
        content = tex_file.read()

    # read text until '% Questions start here' is found
    tex_header = content.split("% Questions start here")[0].strip()

    return tex_header


def parse_footer(file_path):
    """Parse the footer of the LaTeX file to extract the content after the questions end."""
    with open(file_path, "r", encoding="utf-8") as tex_file:
        content = tex_file.read()

    # read text after '% Questions end here'
    tex_footer = content.split("% Questions end here")[1].strip()

    return tex_footer


def parse_questions(file_path, show_questions=False):
    """Parse the questions from the LaTeX file and categorize them."""
    with open(file_path, "r", encoding="utf-8") as tex_file:
        content = tex_file.read()

    # Regex to extract questions and categories
    pattern = (
        r"%-\s*(\w+).*?\n\s*\\item\s(.*?)\n\s*\\begin{enumerate}(.*?)\\end{enumerate}\n"
    )
    question_pattern = re.compile(pattern, re.DOTALL)

    parsed_questions = {}

    for match in question_pattern.findall(content):
        category_header = match[0].strip().split("_")
        category_name = category_header[0]  # Extract the category name
        cols = 2  # Default number of columns
        if len(category_header) > 1:
            cols = category_header[1]  # Extract the number columns (if any)

        question = match[1].strip()
        if show_questions:
            print(f"{category_name}: {question[:50]}")

        options = match[2].split("\\item")[1:]

        if category_name not in parsed_questions:
            parsed_questions[category_name] = []

        parsed_questions[category_name].append((question, options, cols))

    return parsed_questions


def randomize_questions(categorized_questions):
    """Flatten and shuffle the questions."""
    questions_backrefs = []

    # concat all questions in each category
    flat_questions = []
    for _, question_list in categorized_questions.items():
        flat_questions.extend(question_list)

    # shuffle all questions
    shuffled_questions = flat_questions.copy()
    random.shuffle(shuffled_questions)

    for _, question in enumerate(shuffled_questions):
        # store the original order of the questions
        for j, original_question in enumerate(flat_questions):
            if question[0] == original_question[0]:
                questions_backrefs.append(j)
                break

    return shuffled_questions, questions_backrefs


def randomize_options(current_questions, original_questions=None, backref=None):
    """Randomize the order of options for each question."""

    if original_questions is None:
        original_questions = []

    if backref is None:
        backref = []

    random_questions = []

    for qi, (question, options, cols) in enumerate(current_questions):
        correct_answer = (
            original_questions[backref[qi]][1][0] if backref else options[0]
        )
        random.shuffle(options)
        random_questions.append(
            (question, options, options.index(correct_answer), cols)
        )

    return random_questions


def generate_latex(
    output_path, current_questions, tex_header="", tex_footer="", show_answers=False
):
    """Generate the LaTeX file with the questions and options."""

    with open(output_path, "w", encoding="utf-8") as tex_file:
        tex_file.write(tex_header)
        tex_file.write("\n% Questions start here\n")
        for question, options, correct, cols in current_questions:
            tex_file.write(f"\\item {question}\n")
            if int(cols) > 1:
                tex_file.write("\\begin{multicols}{" + str(cols) + "}\n")
            tex_file.write("\t\\begin{enumerate}\n")
            for qi, option in enumerate(options):
                if show_answers and qi == correct:
                    tex_file.write(f"\t\t\\item $\\blacktriangleright$ {option}\n")
                else:
                    tex_file.write(f"\t\t\\item {option}\n")
            tex_file.write("\t\\end{enumerate}\n\n")
            if int(cols) > 1:
                tex_file.write("\\end{multicols}\n")

        tex_file.write(tex_footer)


def generate_answers(output_path, current_questions):
    """Generate the answers file."""

    with open(output_path, "w", encoding="utf-8") as tex_file:
        correct_answers = []
        for _, _, correct, _ in current_questions:
            correct_answers.append(correct)

        tex_file.write(",".join(map(str, correct_answers)))


if __name__ == "__main__":
    # parse from command line
    parser = argparse.ArgumentParser(
        description="Randomize questions and options from a LaTeX file."
    )
    parser.add_argument("--input", required=True, help="Path to the input LaTeX file.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--variations", type=int, default=1, help="Number of variations to generate."
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Generate answers file for each variation.",
    )
    parser.add_argument(
        "--show_questions",
        action="store_true",
        help="Show the parsed questions in console.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output folder."
    )

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
    questions_by_category = parse_questions(
        args.input, show_questions=args.show_questions
    )
    total_questions = sum(len(q) for q in questions_by_category.values())
    all_questions = []
    for category, questions in questions_by_category.items():
        all_questions.extend([(q[0], q[1].copy(), q[2]) for q in questions])
    print(f"Total questions: {total_questions}")

    # keep track of the original questions and their order
    backrefs = []

    # generate specified number of variations
    for i in range(args.variations):
        # randomize the order of the questions
        randomized_questions, br = randomize_questions(questions_by_category)
        backrefs.append(br)

        # randomize the order of the options and store the correct answer index
        randomized_questions = randomize_options(
            randomized_questions, original_questions=all_questions, backref=br
        )

        # replace (VARIANT) int the header and footer for the current variant number
        header_variant = header.replace("(VARIANT)", str(i))
        footer_variant = footer.replace("(VARIANT)", str(i))

        # add the variation number to the output file name
        output_file_name = os.path.join(args.output, os.path.basename(output_file_name))
        generate_latex(
            f"{output_file_name}_{i}.tex",
            randomized_questions,
            tex_header=header_variant,
            tex_footer=footer_variant,
            show_answers=False,
        )

        if args.answers:
            # generate the answers file for the current variant
            generate_latex(
                f"{output_file_name}_{i}_answers.tex",
                randomized_questions,
                tex_header=header_variant,
                tex_footer=footer_variant,
                show_answers=True,
            )

        # store the correct answer index for each question
        generate_answers(f"{output_file_name}_{i}_answers.csv", randomized_questions)

    # store the backrefs for each question in a CSV file
    with open(os.path.join(args.output, "backrefs.csv"), "w", encoding="utf-8") as file:
        for br in backrefs:
            file.write(",".join(map(str, br)) + "\n")
