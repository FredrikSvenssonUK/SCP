#!/usr/bin/env python
 
__author__ = "Ulf Norinder"
__date__ = "21/02/10"


def writeOutListsApp(filePath, dataLists, delimiter = '\t'):
    with open(filePath, 'a') as fd:
        numOfLists = len(dataLists)
        lenOfLists = len(dataLists[0])
        for lineNum in range(lenOfLists):
            tmpString = ""
            for column in range(numOfLists):
                tmpString += str(dataLists[column][lineNum]) + delimiter
            fd.write(tmpString.strip() + "\n")
        fd.flush()

def conformal_p_value(calibration_list, item):
    """
    Return the p-value for a single item given a list of nonconformity 
    scores for the calibration set and the correpsonding score for the item.
    """
    poistion = bisect_right(calibration_list, item) # insert in a list (to the right of tied elements)
    p_value = float(poistion)/(float(len(calibration_list)+1))
    return p_value


import os,sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import cloudpickle
from bisect import bisect_right

parser = ArgumentParser()
parser.add_argument('-i','--infile', help='input training file')
parser.add_argument('-n','--nmodels', type=str, help='number of models (default 20 models)')
parser.add_argument('-m','--mode', type=str, choices=['t','p', 'b'], help='mode: build models, predict new data from models, both build and predict')
parser.add_argument('-s','--sep', type=str, choices=['t','c'], help='file separator: tab or comma')
parser.add_argument('-p','--predfile', help='input prediction file if mode == p')
parser.add_argument('-a','--algo', type=str, choices=['rf','gb'], help='algorithm: RF or GBT')
parser.add_argument('-c','--cptype', type=str, choices=['acp', 'indicp', 'icp', 'indscp', 'scp', 'rscp'], help='mode: type of cp: Aggregated, individual ICP, ICP, individual SCP, SCP, random-SCP')
parser.add_argument('-f','--force', type=str,  help='force e.g second column name to "class" by -f 2, f < 0 to skip')
args = parser.parse_args()

infile = args.infile
nmodels = args.nmodels
mode = args.mode
sep = args.sep
predfile = args.predfile
algo = args.algo
cptype = args.cptype
force = args.force

if infile == None or mode == None or sep == None or algo == None:
    parser.print_help(sys.stderr)
    sys.exit(1)

if mode == 'p' and predfile == None or mode == 'b' and predfile == None:
    parser.print_help(sys.stderr)
    sys.exit(1)

if mode == 't' and predfile != None:
    print("training, erasing the models with a prediction file given? Should mode not be 'p'?")
    sys.exit(1)

force = int(force) - 1

nmodelsmax = 20
if nmodels == None:
    nmodels = nmodelsmax
nmodels = int(nmodels)

if cptype  == 'icp':
    print ("Setting models to 1 for icp")
# kfold needs at least 2 splits. Number of models adjusted later to 1
    nmodels = 2

if mode != 'p':
    for xx in range(1, nmodelsmax+1):
        modelfile2 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) + ".model"
        if os.path.isfile(modelfile2):
            os.remove(modelfile2)

    if sep == 't':
        data = pd.read_csv(infile, sep='\t', header = 0, index_col = None)
    if sep == 'c':
        data = pd.read_csv(infile, sep=',', header = 0, index_col = None)
    if 'name' or 'Name' or 'Molecule name' or 'ID' in data.columns:
        data.rename(columns={'name': 'id', 'Name': 'id', 'Molecule name': 'id', 'ID': 'id'}, inplace=True)

    if force >= 0:
        data = data.rename(columns={ data.columns[force]: "class" })
    if 'class' in data.columns:
        data.rename(columns={'class': 'target'}, inplace=True)
 
    data.loc[data['target'] < 0, 'target'] = 0
    target = data['target'].values
    train = data.drop(['id'], axis=1, errors='ignore')
    train = train.drop(['dataset'], axis=1, errors='ignore')
    train = train.drop(['target'], axis=1, errors='ignore').values
    part1 = int(0.7*len(train))

    if cptype == 'scp' or cptype == 'rscp' or cptype == 'indscp':
        idx = np.random.permutation(int(len(train)))
        calset = idx[part1:]
        trainset = idx[:part1]
        traincal = train[calset]
        targetcal = target[calset]
        train = train[trainset]
        target = target[trainset]

    kf = KFold(n_splits=nmodels, shuffle=True)

    xx = 0
    for test_index, train_index in kf.split(train):
        xx = xx + 1
        if cptype  == 'icp':        
            if xx > 1:
                xx = 1
                break
        print ("Working on model", xx)
 
        print ("incoming trainset",len(train))
        if cptype != 'scp' and cptype != 'rscp' and cptype != 'indscp':
            idx = np.random.permutation(int(len(train)))
#            print (idx)
            trainset = idx[:part1]
            calset = idx[part1:]
            traincal = train[calset]
            targetcal = target[calset]
        if cptype  == 'rscp':
            idx = np.random.permutation(int(len(train)))
            part1 = int(0.7*len(train))
            trainset = idx[:part1]
        if cptype  == 'scp' or cptype  == 'indscp':
            trainset = train_index
        if cptype  == 'indicp':
            trainset = train_index
            print ("trainset after k-fold split",len(trainset))
            np.random.shuffle(trainset)
            print (trainset)
            part1 = int(0.7*len(trainset))
            calset = trainset[part1:]
            trainset = trainset[:part1]
            traincal = train[calset]
            targetcal = target[calset]
 
 
#        print (trainset)
#        print (calset)
        print (cptype , "proper trainset", len(trainset), "calset", len(calset))

        modelfile2 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) + ".model"
        f= open(modelfile2, mode='wb')

        if algo == 'rf':
            mlmethod = RandomForestClassifier(n_estimators = 100)
        if algo != 'gb':
            mlmethod = GradientBoostingClassifier(n_estimators = 100)

        mlmethod.fit(train[trainset], target[trainset])
        cloudpickle.dump(mlmethod, f)
        f.close()

        calibr = mlmethod.predict_proba(traincal)
        calfile0 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) +".cal0"
        calfile1 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) +".cal1"
        f= open(calfile0, mode='wb')
        f2= open(calfile1, mode='wb')
        calibr0 = [x[0] for x in calibr]
        calibr1 = [x[1] for x in calibr]
        prob0 = []
        prob1 = []
        for zz in range(0, len(calibr0)):
            abc = targetcal[zz]
            if abc <= 0.0:
                prob0.append(calibr0[zz])
            if abc > 0.0:
                prob1.append(calibr1[zz])
        calibr0 = np.array(prob0)
        calibr1 = np.array(prob1)

        cloudpickle.dump(calibr0, f)
        f.close()
        cloudpickle.dump(calibr1, f2)
        f2.close()


if mode != 't':
    if sep == 't':
        data = pd.read_csv(predfile, sep='\t', header = 0, index_col = None)
    if sep == 'c':
        data = pd.read_csv(predfile, sep=',', header = 0, index_col = None)
    if 'name' or 'Name' or 'Molecule name' or 'ID' in data.columns:
        data.rename(columns={'name': 'id', 'Name': 'id', 'Molecule name': 'id', 'ID': 'id'}, inplace=True)

    if force >= 0:
        data = data.rename(columns={ data.columns[force]: "class" })
        print(data.head())
    if 'class' in data.columns:
        data.rename(columns={'class': 'target'}, inplace=True)

    data.loc[data['target'] < 0, 'target'] = 0
    labels = data['id']
    ll = len(labels)
    target = data['target'].values
    test = data.drop(['id'], axis=1, errors='ignore')
    test = test.drop(['dataset'], axis=1, errors='ignore')
    test = test.drop(['target'], axis=1, errors='ignore').values

    if cptype  == 'icp':        
        nmodels = 1
    xx = nmodels
    outfile = predfile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) + "sum.csv"
    f2 = open(outfile,'w')
    f2.write('id\tp-value_low_class\tp-value_high_class\tclass\tmodel\n')
    f2.close()

    cal_dict_0 = {}
    cal_dict_1 = {}
    compound_dict_0 = {}
    compound_dict_1 = {}

    for xx in range(1, nmodels+1):
        print ("Reading from model", xx)
        num = [xx]*ll

        modelfile2 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) + ".model"
        f= open(modelfile2, mode='rb')
        mlmethod = cloudpickle.load(f)
        f.close()
        calfile0 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) +".cal0"
        calfile1 = infile +"_"+ str(algo) +"_" + cptype + "_" +str(xx) +".cal1"
        f= open(calfile0, mode='rb')
        f2= open(calfile1, mode='rb')
        calibr0 = cloudpickle.load(f)
        calibr1 = cloudpickle.load(f2)
        f.close()
        f2.close()
        predicted = mlmethod.predict_proba(test)
        predicted0 = [x[0] for x in predicted]
        predicted1 = [x[1] for x in predicted]

        if cptype != 'scp' and cptype != 'rscp':
            calibrsort0 = np.sort(calibr0, axis=None)
            calibrsort1 = np.sort(calibr1, axis=None)
            pos0tot = []
            pos1tot = []

            for yy in range(0, len(predicted0)):
                pos0 = conformal_p_value(calibrsort0, float(predicted0[yy]))
                pos1 = conformal_p_value(calibrsort1, float(predicted1[yy]))
#                print (predicted0[yy], pos0, predicted1[yy], pos1)
                pos0tot.append(pos0)
                pos1tot.append(pos1)
            writeOutListsApp(outfile, [labels, pos0tot,  pos1tot, target, num])

        if cptype == 'scp' or cptype == 'rscp':
            print ("Collecting probs for averaging")
            for yy in range(0, len(calibr0)):
                try:
                    cal_dict_0[yy].append([calibr0[yy]])
                except:
                    cal_dict_0[yy] = ([calibr0[yy]])
            for yy in range(0, len(calibr1)):
                try:
                    cal_dict_1[yy].append([calibr1[yy]])
                except:
                    cal_dict_1[yy] = ([calibr1[yy]])

            for yy in range(0, len(predicted0)):
                try:
                    compound_dict_0[yy].append([predicted0[yy]])
                except:
                    compound_dict_0[yy] = [predicted0[yy]]
            for yy in range(0, len(predicted1)):
                try:
                    compound_dict_1[yy].append([predicted1[yy]])
                except:
                    compound_dict_1[yy] = [predicted1[yy]]

    calibr0 = []
    calibr1 = []
    if cptype == 'scp' or cptype == 'rscp':
        print ("Averaging")
        for key in cal_dict_0:
            calibr0.append(float(np.mean(cal_dict_0[key], axis=0)))
        for key in cal_dict_1:
            calibr1.append(float(np.mean(cal_dict_1[key], axis=0)))
        calibrsort0 = np.sort(calibr0, axis=None)
        calibrsort1 = np.sort(calibr1, axis=None)
        pos0tot = []
        pos1tot = []
        for key in compound_dict_0:
            c0 = np.mean(compound_dict_0[key], axis=0)
            c1 = np.mean(compound_dict_1[key], axis=0)
            pos0 = conformal_p_value(calibrsort0, float(c0))
            pos1 = conformal_p_value(calibrsort1, float(c1))
#            print (predicted0[yy], pos0, predicted1[yy], pos1)
            pos0tot.append(pos0)
            pos1tot.append(pos1)
        num = [1]*ll
        writeOutListsApp(outfile, [labels, pos0tot,  pos1tot, target, num])

print (" - finished\n")

