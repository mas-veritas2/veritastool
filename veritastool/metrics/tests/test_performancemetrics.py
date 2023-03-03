import pickle
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.credit_scoring import CreditScoring
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.fairness import Fairness
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../veritastool/examples/customer_marketing_example'))
sys.path.append(module_path)
import selection, uplift, util
from sklearn.linear_model import LogisticRegression

#Load Credit Scoring Test Data
file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'credit_score_dict.pickle')
input_file = open(file, "rb")
cs = pickle.load(input_file)

#Reduce into two classes
cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
#Model Contariner Parameters
y_true = np.array(cs["y_test"])
y_pred = np.array(cs["y_pred"])
y_train = np.array(cs["y_train"])
p_grp = {'SEX': [[1]], 'MARRIAGE':[[1]]}
up_grp = {'SEX': [[2]], 'MARRIAGE':[[2]]}
x_train = cs["X_train"]
x_test = cs["X_test"]
model_name = "credit_scoring"
model_type = "classification"
y_prob = cs["y_prob"]
model_obj = LogisticRegression(C=0.1)
model_obj.fit(x_train, y_train)

#rejection inference
num_applicants = {'SEX': [3500.0, 5000.0], 'MARRIAGE':[3500.0, 5000.0]}
base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}


#Create Model Container and Use Case Object
container = ModelContainer(y_true, p_grp, model_type, model_name,  y_pred, y_prob, y_train, x_train=x_train, \
                           x_test=x_test, model_object=model_obj, up_grp=up_grp)

#Create Use Case Object
cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                           fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                           tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
# cre_sco_obj.k = 1

import pickle
import numpy as np
import pandas as pd
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.customer_marketing import CustomerMarketing
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.fairness import Fairness
import pytest
import sys

#Load Credit Scoring Test Data
file_prop = os.path.join(project_root, 'veritastool', 'examples', 'data', 'mktg_uplift_acq_dict.pickle')
file_rej = os.path.join(project_root, 'veritastool', 'examples', 'data', 'mktg_uplift_rej_dict.pickle')
input_prop = open(file_prop, "rb")
input_rej = open(file_rej, "rb")
cm_prop = pickle.load(input_prop)
cm_rej = pickle.load(input_rej)

#Model Container Parameters
#Rejection Model
y_true_rej = cm_rej["y_test"]
y_pred_rej = cm_rej["y_test"]
y_train_rej = cm_rej["y_train"]
p_grp_rej = {'isforeign':[[0]], 'isfemale':[[0]],'isforeign-isfemale':'maj_rest'}
x_train_rej = cm_rej["X_train"].drop(['ID'], axis = 1)
x_test_rej = cm_rej["X_test"].drop(['ID'], axis = 1)
y_prob_rej = pd.DataFrame(cm_rej["y_prob"], columns=['CN', 'CR', 'TN', 'TR'])
data = {"FEATURE" :['income', 'noproducts', 'didrespond', 'age', 'isfemale',
       'isforeign'], 
        "VALUE":[0.3, 0.2, 0.15, 0.1, 0.05, 0.03]}

#Propensity Model
y_true_prop = cm_prop["y_test"]
y_pred_prop = cm_prop["y_test"]
y_train_prop = cm_prop["y_train"]
y_prob_prop = pd.DataFrame(cm_prop["y_prob"], columns=['CN', 'CR', 'TN', 'TR'])

PROFIT_RESPOND = 190
COST_TREATMENT =20

model_object_rej = cm_rej['model']
model_name_rej = "custmr_marketing"
model_type_rej = "uplift"
model_object_prop = cm_prop['model']
model_type_prop = "uplift"

#fit the models as it's a pre-requisite for transparency analysis
model_object_rej.fit(x_train_rej,y_train_rej)
model_object_prop.fit(x_train_rej,y_train_prop)

container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_pred_rej, y_prob = y_prob_rej, y_train= y_train_rej, \
                               p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, \
                               model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,\
                               pos_label=['TR', 'CR'], neg_label=['TN', 'CN'])

container_prop = container_rej.clone(y_true = y_true_prop, y_pred = y_pred_prop, y_prob = y_prob_prop, y_train= y_train_prop,\
                                model_object = model_object_prop,  pos_label=['TR', 'CR'], neg_label=['TN', 'CN'])


#Create Use Case Object
cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 80, \
                                  fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", \
                                  perf_metric_name = "expected_profit", fair_metric_name="auto", revenue = PROFIT_RESPOND, \
                                  treatment_cost =COST_TREATMENT, tran_index=[20,40], tran_max_sample=1000, \
                                  tran_pdp_feature= ['age','income'], tran_pdp_target='CR', tran_max_display = 6)
# cm_uplift_obj.k = 1

def test_execute_all_perf():
    # cre_sco_obj._compute_fairness(1)
    cre_sco_obj.evaluate()
    assert cre_sco_obj.perf_metric_obj.result != None
    # cm_uplift_obj._compute_fairness(1)
    cm_uplift_obj.evaluate()
    assert cm_uplift_obj.perf_metric_obj.result != None

def test_translate_metric():
    cre_sco_obj.feature_importance()
    assert cre_sco_obj.feature_imp_values != None
    cm_uplift_obj.feature_importance()
    assert cm_uplift_obj.feature_imp_values != None

def test_compute_wape():
    cre_sco_obj.y_true = [[],[]]
    cre_sco_obj.y_true[0] = np.array(y_true)
    cre_sco_obj.y_true[1] = np.array(y_true)

    cre_sco_obj.y_pred = [[],[]]
    cre_sco_obj.y_pred[0] = np.array(y_pred)
    cre_sco_obj.y_pred[1] = np.array(y_pred)
    result = PerformanceMetrics._compute_wape(cre_sco_obj)
    assert result == 0.2652388797364086
    
def test_compute_mape():
    cre_sco_obj.sample_weight = [[],[]]
    cre_sco_obj.sample_weight[0] = np.array([0.7 for x in range(len(y_pred))])
    cre_sco_obj.sample_weight[1] = np.array([0.7 for x in range(len(y_pred))])
    result = PerformanceMetrics._compute_mape(cre_sco_obj)
    assert result == 366292769692800.5

def test_compute_mape():
    cre_sco_obj.sample_weight = [[],[]]
    np.random.seed(0)

    cre_sco_obj.sample_weight[0] = np.random.random_sample((len(y_pred),))
    cre_sco_obj.sample_weight[1] = np.random.random_sample((len(y_pred),))
    result = PerformanceMetrics._compute_mape(cre_sco_obj)
    assert result == 409709206859836.3
    
# def test__compute_rmse_parity():
    # result = FairnessMetrics._compute_rmse_parity(cre_sco_obj)
    # assert result == (0.16043449543124988, 0.5535316670230406)    

def test__compute_rmse():
    result = PerformanceMetrics._compute_rmse(cre_sco_obj)
    assert result == 0.47128394825897024

def test_expected_selection_rate():
    cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_selection_rate", revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
    cm_uplift_obj.feature_importance()
    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][1],3) == 0.042 

def test_compute_roc_auc_rejection_inference():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant",  perf_metric_name = 'roc_auc',
                           # fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                          num_applicants =num_applicants,  base_default_rate=base_default_rate)
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == -0.025
    
def test_compute_roc_auc():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant",  perf_metric_name = 'roc_auc',
                           # fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                         )
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == -0.025

def test_compute_log_loss():

    cre_sco_obj.y_true = [[],[]]
    cre_sco_obj.y_true[0] = np.array(y_true)
    cre_sco_obj.y_true[1] = np.array(y_true)

    cre_sco_obj.y_prob = [[],[]]
    cre_sco_obj.y_prob[0] = np.array(y_prob)
    cre_sco_obj.y_prob[1] = np.array(y_prob)

    cre_sco_obj.sample_weight = [[],[]]
    cre_sco_obj.sample_weight[0] = np.array([0.7 for x in range(len(y_pred))])
    cre_sco_obj.sample_weight[1] = np.array([0.7 for x in range(len(y_pred))])
    cre_sco_obj.evaluate()
    result = cre_sco_obj.perf_metric_obj.result['perf_metric_values']['log_loss'][0]
    #result = PerformanceMetrics._compute_log_loss(cre_sco_obj)
    assert round(result,3) == 0.612
