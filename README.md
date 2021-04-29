# SCP
Synergy Conformal Prediction applied to Large-Scale Bioactivity Datasets and in Federated Learning

python=3 pandas numpy=1.17.2 cloudpickle scikit-learn
pip install nonconformist==2.1.0
Comment:
Some later versions of numpy (>1.17.2) gives an error deprecation message and seems to give different predicted results for some examples.

Command line for predicting external test set using rscp and 10 models:
conformal_prediction2_types.py -n 10 -m b -s t -a rf -c rscp -f -1 -i trainfile -p testfile 
