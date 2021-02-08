from __future__ import print_function, unicode_literals

from PyInquirer import style_from_dict, Token, prompt
from pprint import pprint
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MaxAbsScaler
import onnxruntime
import matplotlib.pyplot as plt


def plot_sample(Y, x=None, label = None, title=None, save=False, figname=None):

    fig, ax = plt.subplots()
    marker = '.'
    markersize = 1

    if not x:
        if isinstance(Y, np.ndarray):
            x = np.linspace(0, len(Y[0]), len(Y[0]))
            for i in range(len(Y)):
                if label:
                    ax.plot(x, Y[i], marker, label = label[i], markersize=markersize)
                else:
                    ax.plot(x, Y[i], marker, markersize=markersize)
        else:
            x = np.linspace(0, len(Y), len(Y))
            if label:
                ax.plot(x, Y, marker, label=label, markersize=markersize)
            else:
                ax.plot(x, Y, marker, markersize=markersize)
    else:
        if isinstance(Y, np.ndarray):
            for i in range(len(Y)):
                if label:
                    ax.plot(x, Y[i], marker, label=label[i], markersize=markersize)
                else:
                    ax.plot(x, Y[i], marker, markersize=markersize)
        else:
            if label:
                ax.plot(x, Y, marker, label=label, markersize=markersize)
            else:
                ax.plot(x, Y, marker, markersize=markersize)

    fig.show()
    if label:
        plt.legend(loc="best", markerscale=10)
    if title:
        plt.title(title)
    if save:
        plt.savefig(figname + '.png')

    return ax

def inference(model, input):

    session = onnxruntime.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    all_predictions = session.run(None, {input_name: input})[0]
    last_prediction = all_predictions[-1][-1]
    softmaxed_last_prediction = np.exp(last_prediction) / np.sum(np.exp(last_prediction))
    prediction = np.where(softmaxed_last_prediction == np.amax(softmaxed_last_prediction))[0][0]
    return prediction, softmaxed_last_prediction

style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

questions = [

    {
        'type': 'input',
        'name': 'file_path',
        'message': 'Enter the file path or the name of the csv file of your weekly 15 minutes voltage data \n (672 data points, i.e C:\\Users\\FellnerD\\Desktop\\filename.csv or filename.csv or filename): \n',
    },
]

answer = '\\\\dummy\\\\Users\\\\andsoon/'
while not os.path.isfile(os.getcwd() + '\\' + answer) and not os.path.isfile(os.getcwd() + '\\' + answer + '.csv') and not os.path.isfile(answer):
    answer = prompt(questions, style=style)
    answer = str(answer.values()).split("'")[1].replace('\\\\', '\\').replace('/', '')
    if answer == 'exit':
        print('Bye!')
        exit()
    if not os.path.isfile(os.getcwd() + '\\' + answer) and not os.path.isfile(os.getcwd() + '\\' + answer + '.csv') and not os.path.isfile(answer):
        pprint('No file %s found! (to exit enter "exit")' %answer)

try:
    input = pd.read_csv(answer, header=None, sep=';', decimal='.', low_memory=False)
except FileNotFoundError:
    try:
        input = pd.read_csv(os.getcwd() + '\\' + answer, header=None, sep=';', decimal='.', low_memory=False)
    except FileNotFoundError:
        input = pd.read_csv(os.getcwd() + '\\' + answer + '.csv', header=None, sep=';', decimal='.', low_memory=False)

pprint('The first few entries should look like this:')
pprint(pd.read_csv(os.getcwd() + '\\' + 'sample3_mlfct' + '.csv', header=None, sep=';', decimal='.', low_memory=False).head(5))

pprint('Data loaded from %s:' %answer)
pprint('first 5 entries:')
pprint(input.head(5))

X = input
X_zeromean = np.array([x - input.mean() for x in input[0]])                       # deduct it's own mean from every sample
scaler = MaxAbsScaler().fit(X_zeromean)                                 # fit scaler as to scale data between -1 and 1
input = scaler.transform(X_zeromean)
input = input.reshape(1,672,1)
input = input.astype('float32')

PV_noPV = inference('PV_noPV_7day_20k.onnx', input)
if PV_noPV[0] == 1:
    PV = 'The connection point has a PV (probability: %.2f%%)' %np.amax(PV_noPV[1]*100)
    pprint(PV)
    malfunction = inference('malfunctions_in_LV_grid_dataset_7day_20k.onnx', input)
    if malfunction[0] == 1:
        mlfct = 'The PV at the connection point has a malfunction! (probability: %.2f%%)' % np.amax(malfunction[1]*100)
        pprint(mlfct)
    else:
        mlfct = 'The PV at the connection point works as expected. (probability: %.2f%%)' % np.amax(malfunction[1]*100)
        pprint(mlfct)
else:
    PV = 'The connection point has no PV (probability: %.2f%%)' %np.amax(PV_noPV[1]*100)
    pprint(PV)

f = open("outcome.txt", "w")
f.write('The outcomes of the analysis of the voltage data provided in %s are:' %answer)
f.write('\n' + PV)
try:
    f.write('\n' + mlfct)
    plot_sample(X, title = PV + '\n' + mlfct, save = True, figname='outcome')
except NameError:
    plot_sample(X, title=PV, save = True, figname='outcome')
    pass
f.close()
