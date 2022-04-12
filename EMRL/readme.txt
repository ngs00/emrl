Explicit Materials Representation Learning (EMRL)

These python codes contain EMRL and its application.
The codes were implemented in Python 3.7 and PyCharm Community version.

(1) Requirement for running EMRL 
EMRL requires the following packages to run:
PyTorch
PyTorch Geometric
Mendeleev
Pandas
Scikit-learn
XGBoost
tqdm
RDKit
xlrd
openpyxl

(2) Usage
Please execute the exec.py file located in this directory.

(3) Output analysis
Output files will be saved under res directory, which contains three directories described below. 
- emb: Directory to store embedding vectors 
- pred: Direcory to store prediction results  
- trained_model: Directory to store trained prediction models

Sample outputs can be found in res_reference directory.

(4) Dataset availability
Currently, the HOIP dataset is publicly accessible. However, we cannot distribute the dataset obtained from Materials Project (https://materialsproject.org). Rather than providing the dataset we used, we listed the material IDs in the Materials Project same as some other packages. In any case, users can reproduce the results of EMRL by generating a dataset file in the same format as the HOIP dataset.

