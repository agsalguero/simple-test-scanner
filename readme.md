# Simple Test Scanner

This repository contains a set of Python applications designed for generating, scanning, and evaluating printed tests. The applications are built to work with LaTeX files and utilize OpenCV for image processing. Below is a detailed description of each application, its purpose, configuration, and functionality.

## `test.py`

The script generates randomized variations of a LaTeX test file, including randomized questions and options. Questions are extracted from a LaTeX file, and the script can generate multiple variations of the test with different randomizations. `(VARIANT)` is a placeholder for the number of variation being generated (0-based index).  

The script replicates the LaTeX structure. All content before the occurrence of `% Questions start here` and after `% Questions end here` is preserved in the output files. 

The format of is questions has to be as follows:
```latex
  %- web
  \item ¿Ea non consequat velit culpa ut enim sit officia fugiat aliqua? 
  \begin{enumerate}
    \item Aliquip in excepteur aute ea adipisicing proident ut quis..
    \item Commodo qui exercitation fugiat quis sunt id.
    \item Id pariatur ex eiusmod est fugiat dolore incididunt et.
    \item Magna duis id voluptate ad labore et.
  \end{enumerate}
```

The category of the question is defined by the first line, which starts with `%-` and contains the category name. The correct answer should always be the first one in the list. Please respect the LaTeX structure and ensure that the questions are formatted correctly, **including the whitespaces and line breaks**. 

By default, the script generates two columns for the answers. If you want to specify the number of columns, you can do so by adding `_X` to the category name, where `X` is the number of columns. For example, `%- web_3` will generate three columns for the answers.

LaTeX commands might be used in the questions. For example, the following question includes LaTeX commands for including an image and a source code listing:

```latex
%- html_1
  \item Sint eiusmod mollit sint esse commodo sit enim consequat consectetur in est ullamco.
  
  \begin{lstlisting}[language=HTML]
  <!DOCTYPE html>
  <html>
    <head><title>Page Title</title></head>
    <body>
      <h1>This is a Heading</h1>
    </body>
  </html>
  \end{lstlisting}

  \begin{figure}[tb]
    \centering
    \includegraphics[width=0.25\textwidth]{src/1/1}
    \vspace*{-5mm}
    \caption{Tabla HTML.}
    \label{fig:1}
  \end{figure}

  \begin{enumerate}
    \item  \hfill \vspace*{-7mm}
    \begin{lstlisting}[language=HTML]
<td>Row 2, Col 1</td>
    \end{lstlisting}
    \item  \hfill \vspace*{-7mm}
    \begin{lstlisting}[language=HTML]
<td>Row 20, Col 11</td>
    \end{lstlisting}
    \item  \hfill \vspace*{-7mm}
    \begin{lstlisting}[language=HTML]
<td>Row 2, Col 100</td>
    \end{lstlisting}
    \item  \hfill \vspace*{-7mm}
    \begin{lstlisting}[language=HTML]
<td>Row 200, Col 1</td>
    \end{lstlisting}
  \end{enumerate}
```

### **Command line arguments**

`--input`: Path to the input LaTeX file containing the test.

`--seed`: Random seed for reproducibility.

`--variations`: Number of test variations to generate (default: `1`).

`--answers`: Flag to also generate LaTeX files with correct answers for each variation.

`--show_questions`: Flag to display parsed questions in the console.

`--output`: Path to the output folder where generated files will be saved.

Example command:
```bash
python test.py --input test.tex --seed 42 --variations 3 --answers --show_questions --output output_folder
```

## `scanner.py`

The script can read an image or PDF file containing a questionnaire and extract the answers marked by the user. The script uses ArUco markers and OpenCV to process the image and identify the marked answers. Multiple regions containing answers matrix may be given as input. The script can also handle multiple pages in a PDF file. The output is a CSV file containing the extracted answers, which can be evaluated with the `eval.py` script. An image of the processed questionnaire might be generated as well. Below is an example of a processed questionnaire image generated by the script:

![Processed Questionnaire Example](examples/output/student_1.jpg)

The script requires the installation of [Poppler](https://poppler.freedesktop.org/), a PDF rendering library, to handle PDF files. In Windows, a precompiled version of Poppler can be downloaded from [this GitHub repository](https://github.com/oschwartz10612/poppler-windows/releases/). The script will automatically detect the Poppler installation path if it is added to the system PATH variable or if Poppler is located in a subfolder of the script's directory. Please note that some LaTeX distributions include Poppler binaries, so the script may work without any additional installation in those cases.

### **Command line arguments**

`--input`: Path to the input image or PDF file containing the scanned questionnaire/s.

`--region`: Region dimensions in percentage format `top_left_x,top_left_y,bottom_right_x,bottom_right_y,rows`. Can be repeated for multiple regions.

`--thresshold`: Threshold for determining if a question is answered (default: `6`).

`--output`: Path to the output folder where processed images and CSV files will be saved.

Example command:
```bash
python scanner.py --input questionnaire.pdf --region 0.1,0.275,0.24,0.965,20 --region 0.335,0.275,0.475,0.965,20 --output output_folder
```

### **eval.py**

The script evaluates the answers extracted from the scanned questionnaires against the correct answers for each test variant. It calculates a score for each student based on the number of correct and incorrect answers. The results are saved in a CSV file.

### **Command line arguments**

`--input`: Path to the folder containing the student answer files and test answer files.

`--correct`: Score for a correct answer (default: `1.0`).

`--wrong`: Penalty for a wrong answer (default: `-0.25`).

`--output`: Path to the output folder where the results CSV file will be saved.

Example command:
```bash
python eval.py --input answers_folder --correct 1.0 --wrong -0.25 --output results_folder
```