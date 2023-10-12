# EasyClaimsDetection

Welcome to the EasyClaimsDetection repository!

To use this repository, please complete the following steps:

* **Step 1**: Create a virtual environment

* **Step 2**: Activate your virtual environment

* **Step 3**: Install the necessary packages using the following command: 
`pip install -r requirements.txt`

You can now interact with the content of this repository!

If you wish to re-run the experiments, or to see how Probabilistic Bisection Algorithm module is used to annotate data, run the command `jupyter lab` from your command line. You can now navigate to the "experiments" folder and run either the "experiments.ipynb" or "annotation.ipynb" Notebooks.

The folder "data" stores the list of claims used in our paper, "Detecting statements in text: a domain-agnostic few-shot solution", as well as the datasets we used, and the records of the annotations sequences for each experiment involving the PBA. Note that we do not share the BDI-Sen (Pérez et al., 2023): you should contact the authors directly to get the data.

The folder "annotator", on the other hand, contains the code for the BisectionAnnotator class, used while threshold-tuning with the Probabilistic Bisection Algorithm.

