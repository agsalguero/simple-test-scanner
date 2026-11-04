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


def parse_questions(file_path, show_questions=False, content=None):
    """Parse the questions from the LaTeX file (or from the given content) and
    categorize them."""
    if content is None:
        with open(file_path, "r", encoding="utf-8") as tex_file:
            content = tex_file.read()

    # Regex to extract questions and categories
    pattern = (
        r"%-\s*(\w+).*?\n\s*\\item\s(.*?)\n\s*\\begin{enumerate}(.*?)\\end{enumerate}[ \t]*\n"
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


LABEL_PATTERN = re.compile(r"(\\label\{|label=\{)([^}]*)\}")
REF_PATTERN = re.compile(
    r"(\\(?:label|ref|pageref|eqref|autoref|nameref|cref|Cref)\{|label=\{)([^}]*)\}"
)
MACRO_PATTERN = re.compile(r"\\(?:re)?newcommand\*?\{?\\([A-Za-z]+)\}?\s*\{")


def read_braced(text, start):
    """Return (content, end) of the brace group whose opening brace is at
    text[start - 1]."""
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1], i


def make_self_contained(file_path, prefix):
    """Prepare a LaTeX file to be merged with others into a single test.

    - Labels defined in the file (e.g. ``lst:1``) are renamed to
      ``<prefix>-<label>`` in its questions and footer, as well as the references
      to them, so the labels of different files do not collide.
    - Argument-less macros defined in the header of the file (e.g. ``\\Dado``)
      are expanded in its questions, since only one header is kept.

    Returns the transformed content and the appendix of the file: the footer
    content after the ``\\end{enumerate}`` that closes the questions list,
    without ``\\end{document}`` (e.g. code listings referenced by the questions).
    """
    with open(file_path, "r", encoding="utf-8") as tex_file:
        content = tex_file.read()

    header, rest = content.split("% Questions start here", 1)
    questions, footer = rest.split("% Questions end here", 1)

    labels = {m.group(2) for m in LABEL_PATTERN.finditer(questions + footer)}

    def prefix_ref(match):
        if match.group(2) in labels:
            return f"{match.group(1)}{prefix}-{match.group(2)}}}"
        return match.group(0)

    # expand argument-less macros of the header used in the questions
    for match in MACRO_PATTERN.finditer(header):
        name = match.group(1)
        body, _ = read_braced(header, match.end())
        body = REF_PATTERN.sub(prefix_ref, body)
        questions = re.sub(
            r"\\" + name + r"(?![A-Za-z])", lambda _m, b=body: b, questions
        )

    questions = REF_PATTERN.sub(prefix_ref, questions)
    footer = REF_PATTERN.sub(prefix_ref, footer)

    appendix = footer.split("\\end{enumerate}", 1)[-1]
    appendix = appendix.replace("\\end{document}", "").strip()

    content = (
        header
        + "% Questions start here"
        + questions
        + "% Questions end here"
        + footer
    )
    return content, appendix


BLOCK_PATTERN = re.compile(
    r"\\begin\{([A-Za-z]+\*?)\}.*?\\end\{\1\}"
    r"|\\lstinputlisting\s*(?:\[[^\]]*\])?\s*\{[^}]*\}",
    re.DOTALL,
)
REF_ONLY_PATTERN = re.compile(r"\\(?:ref|pageref|autoref|nameref|cref|Cref)\{([^}]*)\}")
REF_RANGE_PATTERN = re.compile(
    r"\\ref\{([^}]*)\}\s*(?:~?-{1,3}~?|\\textendash\s*|a)\s*\\ref\{([^}]*)\}"
)


def split_appendix(appendix):
    """Split an appendix into labeled blocks (environments such as ``lstlisting``
    or ``figure``, or ``\\lstinputlisting`` commands, that define a label).

    Returns (preamble, blocks, trailing): the text before the first labeled block,
    a list of (labels, text) with the text between blocks attached to the next
    block, and the text after the last labeled block."""
    preamble, blocks = None, []
    last = 0
    for match in BLOCK_PATTERN.finditer(appendix):
        labels = [m.group(2) for m in LABEL_PATTERN.finditer(match.group(0))]
        if not labels:
            continue
        text = appendix[last : match.end()]
        if preamble is None:
            preamble = appendix[last : match.start()]
            text = match.group(0)
        blocks.append((labels, text))
        last = match.end()
    if preamble is None:
        return appendix, [], ""
    return preamble, blocks, appendix[last:]


def referenced_labels(texts, appendices):
    """Labels referenced in the given texts. A range of references such as
    ``\\ref{lst:1}-\\ref{lst:6}`` also references the labels of the appendix
    blocks between both labels."""
    ordered = [
        label
        for appendix in appendices
        for labels, _ in split_appendix(appendix)[1]
        for label in labels
    ]
    referenced = set()
    for text in texts:
        for match in REF_ONLY_PATTERN.finditer(text):
            referenced.update(label.strip() for label in match.group(1).split(","))
        for match in REF_RANGE_PATTERN.finditer(text):
            first, last = match.group(1), match.group(2)
            if first in ordered and last in ordered:
                i, j = ordered.index(first), ordered.index(last)
                referenced.update(ordered[min(i, j) : max(i, j) + 1])
    return referenced


def filter_appendix(appendix, referenced):
    """Keep only the labeled blocks of the appendix that are referenced. The text
    before the first block (e.g. an introduction) is kept only if some block is
    kept. Appendices without labeled blocks are kept as they are.

    Returns (filtered appendix, kept blocks, total blocks)."""
    preamble, blocks, trailing = split_appendix(appendix)
    if not blocks:
        return appendix, 0, 0
    kept = [text for labels, text in blocks if referenced.intersection(labels)]
    if not kept:
        return "", 0, len(blocks)
    filtered = preamble + "".join(kept) + trailing
    return filtered.strip(), len(kept), len(blocks)


def randomize_questions(flat_questions):
    """Shuffle the questions and return them along with their backrefs (the index
    of each shuffled question in the original, unshuffled list)."""
    questions_backrefs = list(range(len(flat_questions)))
    random.shuffle(questions_backrefs)
    shuffled_questions = [flat_questions[j] for j in questions_backrefs]

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
        description="Randomize questions and options from one or more LaTeX files."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Path to the input LaTeX file(s). If several files are given, their "
        "questions are merged into a single test (e.g. a global exam built from "
        "several partial exams).",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="LaTeX file whose header and footer are used for the generated tests "
        "(default: the first input file). Also used to name the output files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--variations", type=int, default=1, help="Number of variations to generate."
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions of the generated tests. If lower than the number "
        "of available questions, a random subset is selected (the same for every "
        "variation). Default: all the questions.",
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

    template = args.template if args.template else args.input[0]

    # check if the input files exist
    for input_file in args.input + [template]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            exit(1)

    # create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # set values
    random.seed(args.seed)
    output_file_name = os.path.join(
        args.output, os.path.basename(os.path.splitext(template)[0])
    )

    # extract the header and footer from the template file
    header = parse_header(template)
    footer = parse_footer(template)

    # parse the questions from every input file, keeping the order of the files
    # (questions of the first file, then the second, ...). Backrefs refer to this
    # order, so the scanner reports answers in that same order.
    all_questions = []  # original questions (correct answer is the first option)
    working_questions = []  # questions whose options get shuffled in place
    merge = len(args.input) > 1
    appendices = []
    prefixes = set()
    for input_file in args.input:
        content = None
        if merge:
            if args.show_questions:
                print(f"--- {input_file}")
            # make labels unique and questions independent of the file's header
            prefix = re.sub(
                r"[^A-Za-z0-9]", "", os.path.splitext(os.path.basename(input_file))[0]
            )
            while prefix in prefixes:
                prefix += "x"
            prefixes.add(prefix)
            content, appendix = make_self_contained(input_file, prefix)
            if appendix:
                appendices.append(appendix)
        questions_by_category = parse_questions(
            input_file, show_questions=args.show_questions, content=content
        )
        file_questions = [
            q for questions in questions_by_category.values() for q in questions
        ]
        print(f"{input_file}: {len(file_questions)} questions")
        working_questions.extend(file_questions)
        all_questions.extend([(q[0], q[1].copy(), q[2]) for q in file_questions])

    # warn about repeated questions (e.g. the same question in two partial exams)
    seen = set()
    for question, _, _ in all_questions:
        if question in seen:
            print(f"Warning: repeated question: {question[:50]}")
        seen.add(question)

    print(f"Total questions: {len(all_questions)}")

    # select a random subset of questions (the same for every variation), keeping
    # their original order so backrefs refer to the order of the selected questions
    if args.questions is not None:
        if not 1 <= args.questions <= len(all_questions):
            print(
                f"--questions must be between 1 and {len(all_questions)} "
                f"(got {args.questions})."
            )
            exit(1)
        if args.questions < len(all_questions):
            selected = sorted(random.sample(range(len(all_questions)), args.questions))
            all_questions = [all_questions[j] for j in selected]
            working_questions = [working_questions[j] for j in selected]
            print(f"Selected questions: {len(all_questions)}")
            if args.show_questions:
                for question, _, _ in all_questions:
                    print(f"  {question[:50]}")

    # when merging several files, the footer is the end of the questions list of
    # the template followed by the appendix of the template (if it is not an input
    # file) and the blocks (e.g. code listings) of the appendices of the input
    # files that are referenced by the selected questions
    if merge:
        closing, _, template_appendix = footer.partition("\\end{enumerate}")
        template_appendix = template_appendix.replace("\\end{document}", "").strip()
        merged_appendices = []
        if template_appendix and os.path.abspath(template) not in [
            os.path.abspath(f) for f in args.input
        ]:
            merged_appendices.append(template_appendix)

        referenced = referenced_labels(
            [q[0] + "".join(q[1]) for q in all_questions], appendices
        )
        for appendix in appendices:
            filtered, kept, total = filter_appendix(appendix, referenced)
            if total:
                print(f"Appendix blocks included: {kept}/{total}")
            if filtered:
                merged_appendices.append(filtered)

        footer = (
            closing
            + "\\end{enumerate}\n\n"
            + "\n\n".join(merged_appendices)
            + "\n\n\\end{document}\n"
        )

    # keep track of the original questions and their order
    backrefs = []

    # generate specified number of variations
    for i in range(args.variations):
        # randomize the order of the questions
        randomized_questions, br = randomize_questions(working_questions)
        backrefs.append(br)

        # randomize the order of the options and store the correct answer index
        randomized_questions = randomize_options(
            randomized_questions, original_questions=all_questions, backref=br
        )

        # replace (VARIANT) int the header and footer for the current variant number
        header_variant = header.replace("(VARIANT)", str(i))
        footer_variant = footer.replace("(VARIANT)", str(i))

        # add the variation number to the output file name
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
