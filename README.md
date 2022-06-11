# nps

---

## Cloning a repository

Follow the steps below to clone this repository in your system:

1. Make sure you have **git** installed in your machine. You can install it by opening a terminal session and running the following command: **$ sudo apt install git**
2. You’ll see a "Code" green button on the upper-right side of the repository webpage. Click on it and copy the web URL shown from **HTTPS** tab.
3. Open a terminal instance in your system and navigate to the directory where you want to clone the repository.
4. Write **$git clone **, paste the web URL you copied in step 3 (You can use CTRL+SHIFT+V to paste on the terminal screen) and execute it.
5. Type your Github login and [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).
6. Git should clone the repository: All done!

---

## Prerequisites

You need to install the following packages in your computer:

1. Python3: Most Linux distributions already have Python3 preinstalled
2. PIP3: This is an interface to install Python3 packages. You can install it with: **$ sudo apt install pip3**
3. Numpy, Scipy, and Matplotlib packages for Python3. You can install them with: **$ pip3 install numpy scipy matplotlib**
4. Fortran compiler. You can install it with: **$ sudo apt install gfortran**

---

## Installation instructions

Follow the steps below to install nps in your system:

1. Once you clone the repository, open a terminal session and navigate to the root folder of the repository.
2. Write down the directory that holds the root folder. For instance, if you installed nps in **/home/user/git/nps**, the the directory you must remember is **/home/user/git**.
3. Open your bashrc file with the following command: **$ gedit ~/.bashrc** (You can use another text editor if you wish).
4. Add the following line to the end of your bashrc file: **export PYTHONPATH="${PYTHONPATH}:<dir>"**, where <dir> is the directory you got in step 2. This will include the nps module in Python's search directory, allowing you to import it from any directory in your system.
5. Save and close the text editor.
6. Back into the terminal, type: **$ source ~/.bashrc**. This will reload the definitions from the bashrc file. You may also close and reopen the terminal for these changes to make effect.
7. Still with the terminal at the root folder of the respository, execute the command: **$ make**. Wait until the installation is complete.
8. Test the installation by running **$ make test** in your terminal.

---

## Canonical problems

From the test and other canonical cases, results may differ depending on the versions of the numerical packages.

Regarding the reproducibility of the results shown in the manuscript, each folder has a **.pickle** file, which contains the trained neural network for each case.

To reproduce the main article plots with the trained neural networks, execute the following Python3 codes:
- **/nps/examples/canonical_cases/ann_solution/canonical_problems_plot.py**
- **/nps/examples/pot_flow/potflow_1_ANN_plot.py**
- **/nps/examples/pot_flow/potflow_2_ANN_plot.py**

To retrain the article cases neural networks from the scratch, execute the following Python3 codes:
- **/nps/examples/canonical_cases/ann_solution/canonical_problems.py**
- **/nps/examples/pot_flow/potflow_1_ANN.py**
- **/nps/examples/pot_flow/potflow_2_ANN.py**

---

## Contact

Report any issues to Ney Sêcco at ney@ita.br
