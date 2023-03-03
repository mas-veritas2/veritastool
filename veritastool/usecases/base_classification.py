import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from ..principles import Fairness, Transparency
from ..metrics.fairness_metrics import FairnessMetrics
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.newmetric import *
from ..util.utility import check_datatype, check_value
from ..config.constants import Constants
from ..util.errors import *

class BaseClassification(Fairness, Transparency):
    """
    Class to evaluate and analyse fairness in general classification applications.

    Class Attributes
    ------------------
    _model_type_to_metric_lookup: dictionary
                Used to associate the model type (key) with the metric type, expected size of positive and negative labels (value) & length of model_params respectively.
                
                e.g. {"base_classification": ("classification", 2, 1), “rejection”: (“classification”, 2, 1), “uplift”: (“uplift”, 4, 2), “a_new_type”: (“regression”, -1, 1)}
    """

    _model_type_to_metric_lookup = {"classification": ("classification", 0, 1)}
    _model_data_processing_flag = False

    def __init__(self, model_params, fair_threshold, fair_is_pos_label_fav = True, perf_metric_name = "balanced_acc", fair_metric_name = "auto", fair_concern = "eligible", fair_priority = "benefit", fair_impact = "normal", fair_metric_type = "difference", fairness_metric_value_input = {}, tran_index = [1], tran_max_sample = 1, tran_pdp_feature = [], tran_pdp_target=None, tran_max_display = 10,tran_features=[]):
        """
        Parameters
        ----------
        model_params: list containing 1 ModelContainer object
                Data holder that contains all the attributes of the model to be assessed. Compulsory input for initialization. Single object corresponds to model_type of "default".

        fair_threshold: int or float
                Value between 0 and 100. If a float between 0 and 1 (not inclusive) is provided, it is converted to a percentage and the p % rule is used to calculate the fairness threshold value.
                If an integer between 1 and 100 is provided, it is converted to a percentage and the p % rule is used to calculate the fairness threshold value.

        fair_is_pos_label_fav: boolean, default=True
                Used to indicate if positive label specified is favourable for the classification use case. If True, 1 is specified to be favourable and 0 as unfavourable.
        
        Instance Attributes
        --------------------
        perf_metric_name: string, default='balanced_acc'
                Name of the primary performance metric to be used for computations in the evaluate() and/or compile() functions.

        fair_metric_name : string, default="auto"
                Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions.

        fair_concern: string, default="eligible"
                Used to specify a single fairness concern applied to all protected variables. Could be "eligible" or "inclusive" or "both".

        fair_priority: string, default="benefit"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "benefit" or "harm"

        fair_impact: string, default="normal"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "normal" or "significant" or "selective"
        
        fair_metric_type: str, default='difference'
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "difference" or "ratio"

        fairness_metric_value_input : dictionary
                Contains the p_var and respective fairness_metric and value 
                e.g. {"gender": {"fnr_parity": 0.2}}

        _use_case_metrics: dictionary of lists, default=None
                Contains all the performance & fairness metrics for each use case.
                e.g. {"fair ": ["fnr_parity", ...], "perf": ["balanced_acc, ..."]}
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric above.

        _input_validation_lookup: dict
                Contains the attribute and its correct data type for every argument passed by user. Used to perform the Utility checks.
                e.g. _input_validation_lookup = {
                "fair_threshold": [(float, int), (Constants().fair_threshold_low), Constants().fair_threshold_high],
                "fair_neutral_tolerance": [(float,),(Constants().fair_neutral_threshold_low), Constants().fair_neutral_threshold_high],
                "sample_weight": [(int,), (0, np.inf)],
                "perf_metric_name": [(str,), _use_case_metrics["perf"]],
                "fair_metric_name": [(str,), _use_case_metrics["fair"]],
                "concern": [(str,), ["eligible", "inclusion", "both"]]
                }

        k : int
                Integer from Constants class to calculate confidence interval

        array_size : int
                Integer from Constants class to fix array size

        decimals : int
                Integer from Constants class to fix number of decimals to round off

        err : object
                VeritasError object
        
        e_lift : float, default=None
                Empirical lift

        pred_outcome: dictionary, default=None
                Contains the probabilities of the treatment and control groups for both rejection and acquiring
        """
        Fairness.__init__(self,model_params, fair_threshold, fair_metric_name, fair_is_pos_label_fav, fair_concern, fair_priority, fair_impact, fair_metric_type, fairness_metric_value_input)
        Transparency.__init__(self, tran_index, tran_max_sample, tran_pdp_feature, tran_pdp_target, tran_max_display, tran_features)
        self.perf_metric_name = perf_metric_name
     
        self.e_lift = None
        self.pred_outcome = None
        self.multiclass_flag = False

        if not BaseClassification._model_data_processing_flag:            
            if self.model_params[0].model_object.classes_ is not None and len(self.model_params[0].model_object.classes_)>2 and self.model_params[0].pos_label is None and self.model_params[0].neg_label is None:
                self.classes_ = self.model_params[0].model_object.classes_
                self.multiclass_flag = True                
            else:
                self._model_data_processing()
                BaseClassification._model_data_processing_flag = True
        
        self._check_input()        
        self._check_non_policy_p_var_min_samples()
        self._auto_assign_p_up_groups()
        self.feature_mask = self._set_feature_mask()
        self._tran_check_input()

        if self.multiclass_flag:
                self.mitigate_methods = ['reweigh','correlate']

    def _check_input(self):
        """
        Wrapper function to perform all checks using dictionaries of datatypes & dictionary of values.
        This function does not return any value. Instead, it raises an error when any of the checks from the Utility class fail.
        """

        #check label values in model_params against the usecase's specified model_type info.
        self.check_label_data_for_model_type()

        #check datatype of input variables to ensure they are of the correct datatype
        check_datatype(self)

        #check datatype of input variables to ensure they are reasonable
        check_value(self)

        #check for model_params
        mp_given = len(self.model_params)
        mp_expected = self._model_type_to_metric_lookup[self.model_params[0].model_type][2]
        if mp_given != mp_expected:
            self.err.push('length_error', var_name="model_params", given=str(mp_given), expected=str(mp_expected), function_name="_check_input")

        #check for conflicting input values
        self._base_input_check()

        #check if input variables will the correct fair_metric_name based on fairness tree
        self._fairness_metric_value_input_check()

        # check if y_pred is not None 
        if self.model_params[0].y_pred is None:
            self.err.push('type_error', var_name="y_pred", given= "type None", expected="type [list, np.ndarray, pd.Series]", function_name="_check_input")

        # check if y_prob is float
        if self.model_params[0].y_prob is not None:
            if self.model_params[0].y_prob.dtype.kind == "i":
                self.err.push('type_error', var_name="y_prob", given= "type int", expected="type float", function_name="_check_input")

        #print any exceptions occured
        self.err.pop()

    def _select_fairness_metric_name(self):
        """
        Retrieves the fairness metric name based on the values of model_type, fair_concern, fair_impact, fair_priority, fair_metric_type.
        """
        if self.fair_metric_name == 'auto':
            self.fair_metric_name = self._fairness_tree(self.fair_is_pos_label_fav)
        else :
            self.fair_metric_name

    def _get_confusion_matrix(self, y_true, y_pred, sample_weight, curr_p_var = None, feature_mask = None, **kwargs):
        """
        Compute confusion matrix

        Parameters
        ----------
        y_true : np.ndarray
                Ground truth target values.

        y_pred : np.ndarray
                Copy of predicted targets as returned by classifier.

        sample_weight : array of shape (n_samples,), default=None
                Used to normalize y_true & y_pred.

        curr_p_var : string, default=None
                Current protected variable

        feature_mask : dictionary of lists, default = None
                Stores the mask array for every protected variable applied on the x_test dataset.

        Returns
        -------
        Confusion matrix metrics based on privileged and unprivileged groups or a list of None if curr_p_var == None
        """
        #confusion matrix will only run for classification models        
        if self._model_type_to_metric_lookup[self.model_params[0].model_type][0] == "classification" :

            if 'y_true' in kwargs:
                y_true = kwargs['y_true']

            if 'y_pred' in kwargs:
                y_pred = kwargs['y_pred']

            if curr_p_var is None:
                if y_pred is None:
                    return [None] * 4

                tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight).ravel()
                
                return tp, fp, tn, fn
            else :
                if y_pred is None:
                    return [None] * 8

                mask = feature_mask[curr_p_var] 
                    
                if sample_weight is None :
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true=np.array(y_true)[mask], y_pred=np.array(y_pred)[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(y_true=np.array(y_true)[~mask], y_pred=np.array(y_pred)[~mask]).ravel()
                else :
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true=np.array(y_true)[mask], y_pred=np.array(y_pred)[mask], sample_weight = sample_weight[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(y_true=np.array(y_true)[~mask], y_pred=np.array(y_pred)[~mask], sample_weight = sample_weight[~mask]).ravel()

                return tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u
        else :
            if curr_p_var is None :
                return [None] * 4  
            else :
                return [None] * 8
    
    def _get_sub_group_data(self, grp, perf_metric='sample_count', is_max_bias=True):
                        
        pos_class_count = 0
        neg_class_count = 0
        if is_max_bias:
                metric_val = self.perf_metric_obj.translate_metric(perf_metric, obj=self.perf_metric_obj, subgrp_y_true=grp['y_true'].values, subgrp_y_pred=grp['y_pred'].values) 
        else:
                metric_val = None
        
        return pd.Series([pos_class_count, neg_class_count, metric_val]) 


    def tradeoff(self, output=True, n_threads=1, sigma = 0):
                        
        if self.multiclass_flag:
                print("Tradeoff analysis is not supported for multiclass classification.")
        else:
                
                super(BaseClassification, self).tradeoff( output,n_threads,sigma)

    def feature_importance(self, output=True, n_threads=1, correlation_threshold=0.7):
                        
        if self.multiclass_flag:
                print("Tradeoff analysis is not supported for multiclass classification.")
        else:
                super(BaseClassification, self).feature_importance(output,n_threads,correlation_threshold)

    def explain(self, disable = None, local_index = None, output = True, model_num = None):
                        
        if self.multiclass_flag:
                print("Transparency analysis is not supported for multiclass classification.")
        else:
                super(BaseClassification, self).explain(disable,local_index,output,model_num)

