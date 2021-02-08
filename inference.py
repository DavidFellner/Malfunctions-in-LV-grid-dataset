import onnxruntime
import numpy as np


def inference(model, input):

    session = onnxruntime.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    all_predictions = session.run(None, {input_name: input})[0]
    last_prediction = all_predictions[-1][-1]
    softmaxed_last_prediction = np.exp(last_prediction) / np.sum(np.exp(last_prediction))
    prediction = np.where(softmaxed_last_prediction == np.amax(softmaxed_last_prediction))[0][0]
    return prediction, softmaxed_last_prediction