import numpy as np
from sklearn.metrics import confusion_matrix
from ..principles import Fairness, Transparency
from ..util.utility import check_datatype, check_value
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.fairness_metrics import FairnessMetrics
from ..config.constants import Constants
from ..util.errors import *

class CustomerMarketing(Fairness,Transparency):
    """
    A class to evaluate and analyse fairness in customer marketing related applications.

    Class Attributes
    ------------------
    _model_type_to_metric_lookup: dict
                Used to associate the model type (key) with the metric type, expected size of positive and negative labels (value) & length of model_params respectively.
                e.g. {“rejection”: (“classification”, 2, 1), “uplift”: (“uplift”, 4, 2), “a_new_type”: (“regression”, -1, 1)}

    """
    _model_type_to_metric_lookup = {"uplift":("uplift", 4, 2),
                                   "classification":("classification", 2, 1)}
    _model_data_processing_flag = False

    def __init__(self, model_params, fair_threshold, fair_is_pos_label_fav = True, perf_metric_name =  "expected_profit", fair_metric_name = "auto",  fair_concern = "eligible", fair_priority = "benefit", fair_impact = "normal", fair_metric_type = 'difference', treatment_cost = None, revenue = None, fairness_metric_value_input = {}, proportion_of_interpolation_fitting = 1.0, tran_index = [1], tran_max_sample = 1, tran_pdp_feature = [], tran_pdp_target=None, tran_max_display = 10):
        """
        Parameters
        ----------
        model_params: list 
                Data holder that contains all the attributes of the model to be assessed. Compulsory input for initialization.
                It holds one ModelContainer object(s).
                If a single object is provided, it will be taken as either a "rejection" or "propensity" model according to the model_type flag.
                If 2 objects are provided, while the model_type flag is "uplift", the first one corresponds to rejection model while the second one corresponds to propensity model.
                **x_train[0] = x_train[1] and x_test[0]=x_test[1] must be the same when len(model_param) > 1

        fair_threshold: int or float
                Value between 0 and 100. If a float between 0 and 1 (inclusive) is provided, it is used to benchmark against the primary fairness metric value to determine the fairness_conclusion.
                If an integer between 1 and 100 is provided, it is converted to a percentage and the p % rule is used to calculate the fairness threshold value.

        fair_is_pos_label_fav: boolean, default=True
                Used to indicate if positive label specified is favourable for the classification use case. If True, 1 is specified to be favourable and 0 as unfavourable.

        Instance Attributes
        ------------------
        perf_metric_name: string, default = "balanced_acc"
                Name of the primary performance metric to be used for computations in the evaluate() and/or compile() functions.

        fair_metric_name : string, default = "auto"
                Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions

        fair_concern: string, default = "eligible"
               Used to specify a single fairness concern applied to all protected variables. Could be "eligible" or "inclusive" or "both".

        fair_priority: string, default = "benefit"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "benefit" or "harm"

        fair_impact: string, default = "normal"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "normal" or "significant" or "selective"

        fair_metric_type: str, default='difference'
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "difference" or "ratio"

        treatment_cost: int or float, default=None
                Cost of the marketing treatment per customer

        revenue: int or float, default=None
                Revenue gained per customer

        fairness_metric_value_input : dict
                Contains the p_var and respective fairness_metric and value 
                e.g. {"gender": {"fnr_parity": 0.2}}

        proportion_of_interpolation_fitting : float, default=1.0
                Proportion of interpolation fitting

        _use_case_metrics: dict of list, default=None
                Contains all the performance & fairness metrics for each use case.
                e.g. {"fair ": ["fnr_parity", ...], "perf": ["balanced_acc, ..."]}
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric above.

        _input_validation_lookup: dict
                Contains the attribute and its correct data type for every argument passed by user. Used to perform the Utility checks.
                e.g. _input_validation_lookup = {
                "fair_threshold": [(float,), (int(config.get('threshold','fair_threshold_low')), int(config.get('threshold','fair_threshold_high')))],
                "fair_neutral_tolerance": [(float,) ,(int(config.get('threshold','fair_neutral_tolerance_low')), float(config.get('threshold','fair_neutral_tolerance_high')))],
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

        selection_threshold : float
                Selection threshold from Constants class

        e_lift : float
                Empirical lift

        pred_outcome: dict
                Contains the probabilities of the treatment and control groups for both rejection and acquiring
        """
        #Positive label is favourable for customer marketing use case
        Fairness.__init__(self,model_params, fair_threshold, fair_metric_name, fair_is_pos_label_fav, fair_concern, fair_priority, fair_impact, fair_metric_type, fairness_metric_value_input)
        Transparency.__init__(self, tran_index, tran_max_sample, tran_pdp_feature, tran_pdp_target, tran_max_display)

        # This captures the fair_metric input by the user
        self.fair_metric_input = fair_metric_name
        self.perf_metric_name = perf_metric_name
       
        
        self.proportion_of_interpolation_fitting = proportion_of_interpolation_fitting

    
        self._input_validation_lookup["proportion_of_interpolation_fitting"] = [(float,), (Constants().proportion_of_interpolation_fitting_low), Constants().proportion_of_interpolation_fitting_high]

        self.spl_params = {'revenue': revenue, 'treatment_cost': treatment_cost}
        self.selection_threshold = Constants().selection_threshold

        if not CustomerMarketing._model_data_processing_flag:
            self._model_data_processing()
            CustomerMarketing._model_data_processing_flag = True
        self._check_input()        
        self.e_lift = self._get_e_lift()
        self._check_non_policy_p_var_min_samples()
        self._auto_assign_p_up_groups()
        self.feature_mask = self._set_feature_mask()

        self.pred_outcome = self._compute_pred_outcome()
        self._tran_check_input()

    def _check_input(self):
        """
        Wrapper function to perform all checks using dictionaries of datatypes & dictionary of values.
        This function does not return any value. Instead, it raises an error when any of the checks from the Utility class fail.
        """
        #import error class
        err = VeritasError()        

        #check label values in model_params against the usecase's specified model_type info.
        self.check_label_data_for_model_type()
        
        #check datatype of input variables to ensure they are of the correct datatype
        check_datatype(self)

        #check datatype of input variables to ensure they are reasonable
        check_value(self)

        #check for length of model_params
        mp_g = len(self.model_params)
        mp_e = self._model_type_to_metric_lookup[self.model_params[0].model_type][2]
        if mp_g != mp_e:
            err.push('length_error', var_name="model_params", given=str(mp_g), expected= str(mp_e), function_name="_check_input")

        #check binary restriction for this use case
        if mp_g > 1:
            for i in range(len(self.model_params)):
                self._check_binary_restriction(model_num=i)

        #check for conflicting input values
        self._base_input_check()

        #check if input variables will the correct fair_metric_name based on fairness tree
        self._fairness_metric_value_input_check()

        #check for y_prob not None if model is uplift, else check for y_pred not None
        if self.model_params[0].model_type  == "uplift":
            for i in range(len(self.model_params)):
                if self.model_params[i].y_prob is None:
                    err.push('type_error', var_name="y_prob", given= "type None", expected="type [list, np.ndarray, pd.Series]", function_name="_check_input")
        else:
            for i in range(len(self.model_params)):
                if self.model_params[i].y_pred is None:
                    err.push('type_error', var_name="y_pred", given= "type None", expected="type [list, np.ndarray, pd.Series]", function_name="_check_input")

        #check for y_pred not None if model is uplift, if yes set to None as it is not required
        if self.model_params[0].model_type == 'uplift' and self.model_params[0].y_pred is not None:
            for i in range(len(self.model_params)):
                self.model_params[i].y_pred = None
        
        #check for y_prob is not None, if not None it cannot be an integer
        if self.model_params[0].y_prob is not None:
            if self.model_params[0].y_prob.dtype.kind == "i":
                err.push('type_error', var_name="y_prob", given= "type int", expected="type float", function_name="_check_input")

        #check for revenue and treatment_cost
        #only for uplift models based on expected profit perf metric        
        if self.model_params[0].model_type == 'uplift' and self.perf_metric_name == "expected_profit":
            exp_type = list((int, float))
            spl_range = (0, np.inf)
            #check if spl params are in expected type, otherwise throw exception
            for i in self.spl_params.keys() :
                if type(self.spl_params[i]) not in exp_type :
                    err.push('type_error', var_name=str(i), given=type(self.spl_params[i]), expected=exp_type, function_name="_check_input")
                #check if spl params are within expected range, otherwise throw exception
                try:
                    if type(self.spl_params[i]) != type(None):
                        if  self.spl_params[i] < spl_range[0]  or self.spl_params[i] > spl_range[1] :
                            err.push('value_error', var_name=str(i), given=self.spl_params[i],  expected="range " + str(spl_range), function_name="_check_input")
                except:
                    pass
            #check if in spl params, revenue value provided is not less than treatment_cost 
            try:
                if self.spl_params['revenue'] < self.spl_params['treatment_cost']:
                    err.push('value_error_compare', var_name_a="revenue", var_name_b="treatment_cost", function_name="_check_input")
            except:
                pass
        
        #print any exceptions occured        
        err.pop()
    
    def _get_confusion_matrix(self, y_true, y_pred, sample_weight, curr_p_var = None, feature_mask = None, **kwargs):

        """
        Compute confusion matrix

        Parameters
        ----------
        y_true : numpy.ndarray
                Ground truth target values.

        y_pred : numpy.ndarray
                Copy of predicted targets as returned by classifier.

        sample_weight : numpy.ndarray, default=None
                Used to normalize y_true & y_pred.

        curr_p_var : string, default=None
                Current protected variable

        feature_mask : dict of list, default = None
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
            
            if curr_p_var is None :
                if y_pred is None:
                    return [None] * 4
                
                if sample_weight is None :
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                else :
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight = sample_weight).ravel()
                return tp, fp, tn, fn 
            
            else:
                if y_pred is None:
                    return [None] * 8
                
                mask = feature_mask[curr_p_var]
                if sample_weight is None :
                    mask = self.feature_mask[curr_p_var]
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(np.array(y_true)[~mask], np.array(y_pred)[~mask]).ravel()
                else :
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask], sample_weight = sample_weight[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(np.array(y_true)[~mask], np.array(y_pred)[~mask], sample_weight = sample_weight[~mask]).ravel()
                return tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u
        else :
            if curr_p_var is None:
                return [None] * 4
    
            else:
                return [None] * 8
            
    def _select_fairness_metric_name(self):
        """
        Retrieves the fairness metric name based on the values of model_type, fair_concern, fair_impact, fair_priority and fair_metric_type.
        Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions                
        """
        #if model type is uplift, will not use fairness_tree
        if self.fair_metric_name == 'auto':
            if self.model_params[0].model_type == 'uplift':
                self.fair_metric_name = 'rejected_harm'
            elif self.model_params[0].model_type == 'classification':
                self.fair_metric_name = self._fairness_tree(self.fair_is_pos_label_fav)
        else :
            self.fair_metric_name

    def _get_e_lift(self, **kwargs):
        """
        Computes the empirical lift

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray or None
                Predicted targets as returned by classifier.

        Returns
        -----------
        e_lift : float or None
            Empirical lift value
        """
        #e_lift will only run for uplift models
        if self.model_params[0].model_type == 'uplift':
            
            if 'y_pred_new' in kwargs:
                y_prob = kwargs['y_pred_new']

            else:                
                y_prob = self.model_params[1].y_prob

            y_train = self.model_params[1].y_train

            if y_train is None :
                y_train = self.model_params[1].y_true
                
            classes = np.array(['TR', 'TN', 'CR', 'CN'])
            p_base = np.array([np.mean(y_train == lab) for lab in classes])
            pC = p_base[2] + p_base[3]
            pT = p_base[0] + p_base[1]
            e_lift = (y_prob[:, 0] - y_prob[:, 1]) / pT \
                         + (y_prob[:, 3] - y_prob[:, 2]) / pC
            return e_lift
        else:
            return None
        
        
    def _compute_pred_outcome(self, **kwargs) :
        """
        Computes predicted outcome

        Other parameters
        ---------------
        y_pred_new : numpy.ndarray
                Predicted targets as returned by classifier.

        Returns
        -----------
        pred_outcome : dict
                Contains the probabilities of the treatment and control groups for both rejection and acquiring
        """
        #pred_outcome will only run for uplift models
        if self.model_params[0].model_type == 'uplift':

            y_prob = [model.y_prob for model in self.model_params]
            y_train = [model.y_train  if model.y_train is not None else model.y_true for model in self.model_params]        
                        
            if 'y_pred_new' in kwargs:
                y_prob = kwargs['y_pred_new']

            if y_prob[0] is None or y_prob[1] is None:
                return None
            
            classes = np.array(['TR', 'TN', 'CR', 'CN'])
            model_alias = ['rej_', 'acq_']
            pred_outcome = {}
            
            for i in range(len(self.model_params)) :
                y_prob_temp = y_prob[i]
                y_train_temp = y_train[i]
                p_base = np.array([np.mean(y_train_temp == lab) for lab in classes])
                pC = p_base[2] + p_base[3]
                pT = p_base[0] + p_base[1]
                pOcT = y_prob_temp[:, 0] / pT
                pOcC = y_prob_temp[:, 2] / pC
                pred_outcome[model_alias[i] + 'treatment'] = pOcT
                pred_outcome[model_alias[i] + 'control'] = pOcC
            return pred_outcome

        else :
            return None

    def _check_label(self, y, pos_label, neg_label=None, obj_in=None, y_pred_flag=False):
        """
        Creates copy of y_true as y_true_bin and convert favourable labels to 1 and unfavourable to 0 for non-uplift models.
        Overwrites y_pred with the conversion, if `y_pred_flag` is set to True.
        Checks if pos_labels are inside y

        Parameters
        -----------
        y : numpy.ndarray
                Ground truth target values.

        pos_label : list
                Label values which are considered favorable.
                For all model types except uplift, converts the favourable labels to 1 and others to 0.
                For uplift, user is to provide 2 label names e.g. [["a"], ["b"]] in fav label. The first will be mapped to treatment responded (TR) & second to control responded (CR).

        neg_label : list, default=None
                Label values which are considered unfavorable.
                neg_label will only be used in uplift models.
                For uplift, user is to provide 2 label names e.g. [["c"], ["d"]] in unfav label. The first will be mapped to treatment rejected (TR) & second to control rejected (CR).

        obj_in : object, default=None
                The object of the model_container class.

        y_pred_flag : boolean, default=False
                Flag to indicate if the function is to process y_pred.

        Returns
        -----------------
        y_bin : list
                Encoded labels.

        pos_label2 : list
                Label values which are considered favorable.
        """
        # uplift model
        # 0, 1 => control (others, rejected/responded)
        # 2, 3 => treatment (others, rejected/responded)
        err = VeritasError()
        err_= []
        model_type = obj_in.model_type

        if neg_label is not None and model_type == 'uplift':
            y_bin = y
            n=0

            row = y_bin == pos_label[0]  
            indices_pos_0 = [i for i, x in enumerate(y_bin) if x == pos_label[0]]
            n += np.sum(row)

            row = y_bin == pos_label[1]  
            indices_pos_1 = [i for i, x in enumerate(y_bin) if x == pos_label[1]]
            n += np.sum(row)

            row = y_bin == neg_label[0]  
            indices_neg_0 = [i for i, x in enumerate(y_bin) if x == neg_label[0]]
            n += np.sum(row)

            row = y_bin == neg_label[1]  
            indices_neg_1 = [i for i, x in enumerate(y_bin) if x == neg_label[1]]
            n += np.sum(row)     

            for i in indices_pos_0:
                y_bin[i] = "TR"
            for i in indices_pos_1:
                y_bin[i] = "CR"
            for i in indices_neg_0:
                y_bin[i] = "TN"
            for i in indices_neg_1:
                y_bin[i] = "CN"

            if n != len(y_bin):
                err_.append(['conflict_error', "pos_label, neg_label", "inconsistent values", pos_label + neg_label])
                for i in range(len(err_)):
                    err.push(err_[i][0], var_name_a=err_[i][1], some_string=err_[i][2], value=err_[i][3],
                            function_name="_check_label")
            pos_label2 = [['TR'],['CR']]
            
        if y_bin.dtype.kind in ['i']:
            y_bin  = y_bin.astype(np.int8)

        err.pop()

        return y_bin, pos_label2