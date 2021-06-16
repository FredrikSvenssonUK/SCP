#SCP
This repository accompanies the publication ”Synergy Conformal Prediction applied to Large-Scale Bioactivity Datasets and in Federated Learning”

Recommended packages:
python=3 pandas numpy=1.17.2 cloudpickle scikit-learn
pip install nonconformist==2.1.0

Comment:
Some later versions of numpy (>1.17.2) gives an error deprecation message and seems to give different predicted results for some examples.


##Usage
```python
conformal_prediction2_types.py [-h] [-i INFILE] [-n NMODELS]
                                      [-m {t,p,b}] [-s {t,c}] [-p PREDFILE]
                                      [-a {rf,gb}]
                                      [-c {acp,indicp,icp,indscp,scp,rscp}]
                                      [-f FORCE]

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        input training file
  -n NMODELS, --nmodels NMODELS
                        number of models (default 20 models)
  -m {t,p,b}, --mode {t,p,b}
                        mode: build models, predict new data from models, both
                        build and predict
  -s {t,c}, --sep {t,c}
                        file separator: tab or comma
  -p PREDFILE, --predfile PREDFILE
                        input prediction file if mode == p
  -a {rf,gb}, --algo {rf,gb}
                        algorithm: RF or GBT
  -c {acp,indicp,icp,indscp,scp,rscp}, --cptype {acp,indicp,icp,indscp,scp,rscp}
                        mode: type of cp: Aggregated, individual ICP, ICP,
                        individual SCP, SCP, random-SCP
  -f FORCE, --force FORCE
                        force e.g second column name to "class" by -f 2, f < 0
                        to skip
```

Command line for predicting external test set using rscp and 10 models:
conformal_prediction2_types.py -n 10 -m b -s t -a rf -c rscp -f -1 -i trainfile -p testfile 

conformal_prediction2_types.py -n 10 -m b -s t -a rf -c rscp -f -1 -i example_data_train.txt -p example_data_test.txt
