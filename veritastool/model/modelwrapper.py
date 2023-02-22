import numpy as np
import pandas as pd

class ModelWrapper(object):
    """
    Abstract Base class to provide an interface that supports non-pythonic models.
    Serves as a template for users to define the model_object
    """

    def __init__(self, model_obj = None, model_file = None, output_file = None,classes=[]):
        """
        Instance attributes
        ----------
        model_obj : object, default=None
                Model object

        model_file : str, default=None
                Path to the model file. e.g. "/home/model.pkl"

        output_file : str, default=None
                Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        """
        self.model_obj = model_obj
        self.model_file = model_file
        self.output_file = output_file
        self.classes_ = classes

    def fit(self, x_train, y_train):
        """
        This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
        An example is as follows:
    
        train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
        import subprocess
        process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        Parameters
        -----------
        x_train: pandas.DataFrame or str
                Training dataset. 
                m_samples refers to number of rows in the training dataset where shape is (m_samples, n_features)
                The string refers to the dataset path acceptable by the model (e.g. HDFS URI).

        y_train : numpy.ndarray 
                Ground truth for training data where length is m_samples
        """
        pass

    def predict(self, x_test):
        """
        This function is a template for user to specify a custom predict() method
        that uses the model saved in self.model_file to make predictions on the test dataset.
    
        Predictions can be either probabilities or labels.
    
        An example is as follows:
    
        pred_cmd = "pred_func --predict {self.model_file} {x_test} {self.output_file}"
        import subprocess
        process = subprocess.Popen(pred_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return pd.read_csv(output_file) // any preprocessing required has to be done too

        Parameters
        -----------
        x_test : pandas.DataFrame or str
                Testing dataset where shape is (n_samples, n_features)
                The string refers to the dataset path acceptable by the model (e.g. HDFS URI).
        Returns
        ----------
        y_pred: list, np.ndarray or pd.Series
                Predictions of model_obj on x_test                  
        """
        pass


    def predict_proba(self, x):
        """
        This function is a template for user to specify a custom predict_proba() method
        that uses the model saved in self.model_file to get probabilities on the given dataset.

        Parameters
        -----------
        x : pandas.DataFrame or str
                Testing dataset where shape is (n_samples, n_features)
                The string refers to the dataset path acceptable by the model (e.g. HDFS URI).
        Returns
        ----------
        y_prob: list, np.ndarray, pd.Series, pd.DataFrame
                Prediction probabilities of model_obj on x_test    
        """
        pass

    def _sampling(self, x_train, y_train, x_test, y_test, mode='per_label', size=100):

        if mode == "first":
            x_train_sample = x_train[:size]
            y_train_sample = y_train[:size]
            x_test_sample = x_test[:size]
            return x_train_sample, y_train_sample, x_test_sample

        elif mode == "per_label":            
            x_train_sample = x_train.groupby(y_train).head(size)
            y_train_sample = np.array(pd.DataFrame(y_train).groupby(y_train).head(size)[0])
            x_test_sample = x_test.groupby(y_test).head(size)
            return x_train_sample, y_train_sample, x_test_sample
        else:
            pass

    def check_fit_predict(self, x_train, y_train, x_test, y_test, mode='per_label', size=100):
        
        x_train_sample, y_train_sample, x_test_sample = self._sampling(x_train, y_train, x_test, y_test, mode, size)
        model = self.model_obj

        try:
            model.fit(x_train_sample,y_train_sample)
            print("Fit Success")
        except Exception as error:
            print("Error during fit ", error)
            return 0

        try:
            model.predict(x_test_sample)
            print("Predict success")                
        except Exception as error:
            print("Error during predict ", error)
            return 0

        try:
            model.predict_proba(x_test_sample)
            print("Predict proba success")
            return 1
        except Exception as error:
            print("Error during predict ", error)
            return 0                


