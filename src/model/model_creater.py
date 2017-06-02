from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import spacy
import numpy as np
MAX_SEQUENCE_LENGTH=30
try:
    import cPickle as pickle
except:
    import pickle


class ModelOperations(object):
    """
    """

    def __init__(self):
        self.nlp = spacy.load('en')

    def load_model(self, json_path, weights_path):
        try:
            model = model_from_json(open(json_path).read())
            model.load_weights(weights_path)
            return model
        except:
            raise Exception('Failed to load model/weights')

    def save_model(self, model, json_path, weights_path):
        """
        Helper wrapper over savemodels and saveweights to help keras dump
        the weights and configuration
        """
        json_string = model.to_json()
        with open(json_path, 'w') as f:
            f.write(json_string)
        model.save_weights(weights_path)


class Predictor(object):

    def __init__(self, json_path, weights_path, normalized_x, normalized_y,
                 **kwargs):

        modoperations = ModelOperations()
        self.model = modoperations.load_model(json_path, weights_path)

    def compile_model(self, loss, optimizer, **kwargs):
        """
        Similar to Keras compile function
        Expects atleast losstype and optimizer.
        """
        self.model.compile(loss=loss, optimizer=optimizer, **kwargs)

    def preprocess_data(self,input1,input2):
        in1 = self.clean_data(input1)
        in2 = self.clean_data(input2)
        in1 = self.padd_sequence(in1)
        in2 = self.padd_sequence(in2)
        return in1,in2
    def clean_data(self,input):
        return [x.vector for x in nlp(unicode(input)) if not x.is_stop]

    def padd_sequence(self,input):
        arr = np.zeros(300,MAX_SEQUENCE_LENGTH)
        for idx, x in enumerate(input):
            if idx<MAX_SEQUENCE_LENGTH:
                arr[idx]= x
        return arr
    def predict(self, input_1,input_2):
        """
        Make predictions, given some input data
        This normalizes the predictions based on the real normalization
        parameters and then generates a prediction
        :param X_input
            input vector to for prediction
        """
        in1,in2 = self.preprocess_data(input_1,input_2)
        return self.model.predict(in1,in2)
