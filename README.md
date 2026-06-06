# SPES-HES-HGN-MESM

This repository provides the code, database, trained model objects, and graphical user interface for the Membrane Ensemble Surrogate Model (MESM) developed for SPES/HES/HGN hybrid membrane performance prediction.

The MESM predicts permeate flux and antibiotic rejection as functions of operating pressure, feed concentration, and pH. The model uses a multi-input multi-output neural-network ensemble with ReLU, tanh, and logistic base learners, followed by a meta-layer. The workflow includes z-score standardization, nested leave-one-out cross-validation, Optuna-based hyperparameter optimization, uncertainty estimation, SHAP-based interpretation, response-grid generation, and GUI-compatible model export.

Repository contents

Database.xlsx
MESM_training.py
Membrane Ensemble Surrogate Model.py
requirements.txt
relu_model.pkl
tanh_model.pkl
logistic_model.pkl
meta_layer_model.pkl
README.md
LICENSE

Model inputs and outputs

The MESM uses three operating inputs:

Pressure
Conc
pH

The model predicts five membrane-performance outputs:

Flux
SMX rejection
TRM rejection
TET rejection
ERY rejection

The valid interpolation domain is:

Pressure: 0.5-3.5 bar
Conc: 2-10 mg/L
pH: 5-9

Predictions outside this range should be treated as extrapolations.

For installation, create a fresh Python environment and install the required packages:

pip install -r requirements.txt
or
pip install numpy pandas scikit-learn optuna openpyxl scipy matplotlib shap customtkinter joblib

A typical requirements.txt file should include:

numpy
pandas
scikit-learn
optuna
openpyxl
scipy
matplotlib
shap
customtkinter

Run the full manuscript-consistent training, validation, interpretation, and export workflow using:

python MESM_training.py --database Database.xlsx --output-dir MESM_outputs --compute-shap

The default number of Optuna trials is 25,000 per activation per outer leave-one-out fold, matching the reported MESM workflow.

After execution, the script writes the trained model files to:

MESM_outputs/models/

and the reproducibility tables to:

MESM_outputs/tables/

The exported files include cleaned data, selected hyperparameters, outer-fold predictions, full-fit predictions, model metrics, meta-layer weights, scaling statistics, permutation importance, SHAP importance, SHAP values, and response-grid predictions.

The four files required by the GUI are:

relu_model.pkl
tanh_model.pkl
logistic_model.pkl
meta_layer_model.pkl

For running the GUI, place the four .pkl files in the same folder as the GUI script, then run:

python Membrane Ensemble Surrogate Model.py

The GUI allows interactive prediction of flux and antibiotic rejection, visualization of operating-space responses, uncertainty display, extrapolation warning, and multi-objective optimization inside the experimental domain.


If you use this code or database, please cite the associated manuscript:
Interfacial nanostructure engineering of holey graphene-eutectic solvent hybrid membranes with data-driven surrogate modeling for advanced water purification.


License

This repository is released under the MIT License.
