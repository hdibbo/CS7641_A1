README.txt

Title: Instructions for Running the LaTeX and Machine Learning Code of CS7641 Assignment 1

1. Overview
------------
This project contains a LaTeX document for the classification of Adverse Drug Reactions (ADRs) and medication adherence analysis, as well as a Python script for training machine learning models to predict Adverse Drug Reactions (ADRs) and medication adherence analysis.

2. Prerequisites
----------------
Ensure that you have the following installed on your system:

For LaTeX:
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- An editor to write and compile LaTeX (e.g., TeXShop, Overleaf, TeXworks, Visual Studio Code with LaTeX Workshop extension)

For Python:
- Python 3.6 or higher
- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib

You can install the required Python packages using pip: pip install numpy pandas scikit-learn imbalanced-learn matplotlib


3. Files Included
------------------
- README.txt: This file, providing instructions for running the LaTeX and Python code.
- dhur8-analysis.tex READ ONLY link and its pdf: The LaTeX source file containing the document content 
(https://www.overleaf.com/read/pjbjytxjkkzf#ac8705).
- 'adr.py' and 'adherence.py': The Python scripts containing the machine learning code 
(https://gatech.box.com/s/vb3y5ffie4k22iqscbqz7wum5hgztcxy).
- 'adr_dataset.csv' and `medication_adherence.csv` : The dataset file for the Python script 
(https://gatech.box.com/s/vb3y5ffie4k22iqscbqz7wum5hgztcxy).



4. Instructions for LaTeX
--------------------------
Follow these steps to compile the LaTeX document and generate a PDF:

Step 1: Open your LaTeX editor.

Step 2: Copy the READ ONLY link (https://www.overleaf.com/read/pjbjytxjkkzf#ac8705) of dhur8-analysis.tex file into your LaTeX editor.

Step 3: View dhur8-analysis.tex file.



5. Instructions for Python
--------------------------
Follow these steps to run the Python script and generate the machine learning model outputs:

Step 1: Ensure all required packages are installed. You can do this by running the command:

Step 2: open the provided Folder containing Python codes ('adr.py' and 'adherence.py') at Gatech BOX app 
(https://gatech.box.com/s/vb3y5ffie4k22iqscbqz7wum5hgztcxy).

Step 3: Ensure the dataset Folder 'adr_dataset.csv' and `medication_adherence.csv` are in the same Folder
(https://gatech.box.com/s/vb3y5ffie4k22iqscbqz7wum5hgztcxy).

Step 4: Ensure the Folder where the 'adr.py' and 'adherence.py' files are saved.

Step 5: Run the Python script that load the dataset, train machine learning models, and print the classification reports. Also, it will plot learning curves and validation curves for model performance evaluation.



6. Outputs
-----------
The LaTeX script generates a PDF document with the analysis and results.

The Python script generates the following outputs:
- Classification reports for each machine learning model.
- Learning curves for Neural Network, SVM with Linear Kernel, SVM with RBF Kernel, and k-Nearest Neighbors.
- Validation curves for Neural Network, SVM with Linear Kernel, SVM with RBF Kernel, and k-Nearest Neighbors.


These outputs will be displayed in the terminal and as plots in separate windows.


7. Contact Information
-----------------------
For further assistance, please contact:
- Name: David Hur
- davidhur@gatech.edu






