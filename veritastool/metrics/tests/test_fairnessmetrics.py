import pickle
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.credit_scoring import CreditScoring
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.fairness import Fairness
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../veritastool/examples/customer_marketing_example'))
sys.path.append(module_path)
import selection, uplift, util

#Load Credit Scoring Test Data
file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'credit_score_dict.pickle')
input_file = open(file, "rb")
cs = pickle.load(input_file)
input_file.close()

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
model_obj = cs["model"]
model_obj.fit(x_train, y_train)

#rejection inference
num_applicants = {"SEX": [5841,5841], "MARRIAGE": [5841,5841]}
base_default_rate = {"SEX": [0.5,0.5], "MARRIAGE": [0.5,0.5]}

#Create Model Container 
container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                           x_test=x_test, model_object=model_obj, up_grp=up_grp)

#Create Use Case Object
cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant", 
                           num_applicants =num_applicants,  base_default_rate=base_default_rate,
                           tran_index=[20,40], tran_max_sample = 10, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)

import pickle
import numpy as np
import pandas as pd
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.customer_marketing import CustomerMarketing
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.fairness import Fairness
import pytest

#Load Customer Marketing Test Data
file_prop = os.path.join(project_root, 'veritastool', 'examples', 'data', 'mktg_uplift_acq_dict.pickle')
file_rej = os.path.join(project_root, 'veritastool', 'examples', 'data', 'mktg_uplift_rej_dict.pickle')
input_prop = open(file_prop, "rb")
input_rej = open(file_rej, "rb")
cm_prop = pickle.load(input_prop)
cm_rej = pickle.load(input_rej)

input_prop.close()
input_rej.close()

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
COST_TREATMENT = 20

model_object_rej = cm_rej['model']
model_name_rej = "custmr_marketing"
model_type_rej = "uplift"
model_object_prop = cm_prop['model']
model_type_prop = "uplift"

#fit the models as it's a pre-requisite for transparency analysis
model_object_rej.fit(x_train_rej, y_train_rej)
model_object_prop.fit(x_train_rej, y_train_prop)

#Create Model Containers 
container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_pred_rej, y_prob = y_prob_rej, y_train= y_train_rej, \
                               p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, \
                               model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,\
                               pos_label = ['TR', 'CR'], neg_label = ['TN', 'CN'])

container_prop = container_rej.clone(y_true = y_true_prop, y_pred = y_pred_prop, y_prob = y_prob_prop, y_train= y_train_prop,\
                                model_object = model_object_prop, pos_label = ['TR', 'CR'], neg_label = ['TN', 'CN'])

#Create Use Case Object
cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 80, \
                                  fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", \
                                  fair_metric_name="auto", revenue = PROFIT_RESPOND, \
                                  treatment_cost =COST_TREATMENT, tran_index=[20,40], tran_max_sample=10, \
                                  tran_pdp_feature= ['age','income'], tran_pdp_target='CR', tran_max_display = 6)

#Load Base Regression Test Data
file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'regression_dict.pickle')
input_file = open(file, "rb")
br = pickle.load(input_file)
input_file.close()
#Model Container Parameters
x_train = br["x_train"]
x_test = br["x_test"]
y_train = np.array(br["y_train"])
y_true = np.array(br["y_test"])
y_pred = np.array(br["y_pred"])
p_grp = {'sex': [[1]], 'children': 'maj_min'}

from sklearn.linear_model import LinearRegression
model_object = LinearRegression()
model_name = "base_regression"
model_type = "regression"

#fit the model for fairness diagnosis and transparency assessment
model_object.fit(x_train,y_train)

from veritastool.usecases.base_regression import BaseRegression

#Create Model Container 
container_br = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_train=y_train, x_train=x_train, \
                           x_test=x_test, model_object=model_object)
#Create Use Case Object
base_reg_obj= BaseRegression(model_params = [container_br], fair_threshold = 80, perf_metric_name = "mape", \
                             fair_concern = "eligible", fair_priority = "benefit", fair_impact = "normal", \
                             tran_index = [1,10,25], tran_max_sample = 10, tran_pdp_feature = ['age','bmi'])


import pickle
import numpy as np
import pandas as pd
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.predictive_underwriting import PredictiveUnderwriting
from veritastool.metrics.fairness_metrics import FairnessMetrics
import pytest
import os
from sklearn.metrics import roc_auc_score, log_loss

#Load Predictive Underwriting Test Data
file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'underwriting_dict.pickle')
input_file = open(file, "rb")
puw = pickle.load(input_file)
input_file.close()

#Model Contariner Parameters
y_true = np.array(puw["y_test"])
y_pred = np.array(puw["y_pred"])
y_train = np.array(puw["y_train"])
p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race':'max_bias'}
up_grp = {'gender': [[0]], 'race': [[2, 3]] }
x_train = puw["X_train"]
x_test = puw["X_test"]
y_prob = puw["y_prob"]
model = puw["model"]
model_name = "pred_underwriting"
model_type = "classification"

#Create Model Container 
container_puw = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                           x_test=x_test, model_object=model, up_grp=up_grp)

#Create Use Case Object
pred_underwriting_obj = PredictiveUnderwriting(model_params = [container_puw], fair_threshold = 80, fair_concern = "inclusive", \
                                              fair_priority = "benefit", fair_impact = "normal", fair_metric_type='difference', \
                                              tran_index=[1,2,3], tran_max_sample = 10, tran_max_display = 10, \
                                              tran_pdp_feature = ['age','payout_amount'])

# Setup fixture to test ratio metrics
@pytest.fixture
def ratio_metrics_setup():
    p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race':'max_bias'}
    up_grp = {'gender': [[0]], 'race': [[2, 3]] }
    container_puw = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test, model_object=model, up_grp=up_grp)
    pred_underwriting_obj = PredictiveUnderwriting(model_params = [container_puw], fair_threshold = 80, fair_concern = "inclusive", \
                                                fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio', \
                                                tran_index=[1,2,3], tran_max_sample = 10, tran_max_display = 10, \
                                                tran_pdp_feature = ['age','payout_amount'])
    pred_underwriting_obj.evaluate(output=False)
    
    yield pred_underwriting_obj

# Setup fixture to test multi-class classification
from veritastool.usecases.base_classification import BaseClassification
@pytest.fixture
def multi_class_setup():
    x_train_prop = cm_prop["X_train"].drop(['ID'], axis = 1)
    x_test_prop = cm_prop["X_test"].drop(['ID'], axis = 1)
    y_pred_prop = model_object_prop.predict(x_test_prop)
    p_grp_prop = {'isforeign':[[0]], 'isfemale':[[0]],'isforeign-isfemale':'maj_rest'}
    model_type_prop = 'classification'
    model_name_prop = 'base_classification'
    container_clf = ModelContainer(y_true_prop, p_grp_prop, model_type_prop, model_name_prop, y_pred_prop, y_prob_prop, y_train_prop, \
                            x_train=x_train_prop, x_test=x_test_prop, model_object=model_object_prop, \
                            pos_label=None, neg_label=None) 
    yield container_clf

def test_execute_all_fair():
    # cre_sco_obj._compute_fairness(1)
    cre_sco_obj.evaluate()
    assert cre_sco_obj.fair_metric_obj.result != None
    # cm_uplift_obj._compute_fairness(1)
    cm_uplift_obj.evaluate()
    assert cm_uplift_obj.fair_metric_obj.result != None
    # pred_underwriting_obj._compute_fairness(1)
    pred_underwriting_obj.evaluate()
    assert pred_underwriting_obj.fair_metric_obj.result != None
    base_reg_obj.evaluate()
    assert base_reg_obj.fair_metric_obj.result != None

# def test_translate_metric():
#     cre_sco_obj.feature_importance()
#     assert cre_sco_obj.feature_imp_values != None
#     cm_uplift_obj.feature_importance()
#     assert cm_uplift_obj.feature_imp_values != None

def test_compute_wape_parity():
    expected_without_mask = -0.009    
    result_without_mask = base_reg_obj.fair_metric_obj.result['sex']['fair_metric_values']['wape_parity'][0]    
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    
def test_compute_mape_parity():
    expected_without_mask = 0.074    
    result_without_mask = base_reg_obj.fair_metric_obj.result['sex']['fair_metric_values']['mape_parity'][0]    
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    
def test__compute_rmse_parity():
    expected_with_mask = 4128.476   
    result_with_mask = base_reg_obj.fair_metric_obj.result['children']['fair_metric_values']['rmse_parity'][0]    
    assert round(result_with_mask, 3) == round(expected_with_mask, 3)  


def test_loco_compute_benefit_from_acquiring():
    cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", fair_metric_name = 'acquire_benefit', revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
    cm_uplift_obj.feature_importance(disable=['correlation'])
    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][1],3) == 0.006

def test_loco_compute_disparate_impact_rejection_inference():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", \
                           fair_priority = "benefit", fair_impact = "significant", \
                           fair_metric_name = 'disparate_impact', fair_metric_type='ratio', \
                           num_applicants =num_applicants,  base_default_rate=base_default_rate, \
                           tran_index=[20,40], tran_max_sample = 10, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10, \
                           )
    cre_sco_obj.feature_importance(disable=['correlation'])
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == -0.105
    
def test_loco_compute_disparate_impact():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", \
                           fair_priority = "benefit", fair_impact = "significant", \
                           fair_metric_name = 'disparate_impact', fair_metric_type='ratio', \
                           tran_index=[20,40], tran_max_sample = 10, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    cre_sco_obj.feature_importance(disable=['correlation'])
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == -0.071

def test_compute_positive_predictive_parity():
    # test without feature_mask
    expected_without_mask = 0.015
    result_without_mask = pred_underwriting_obj.fair_metric_obj.result['gender']['fair_metric_values']['ppv_parity'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = -0.017
    result_mask = pred_underwriting_obj.fair_metric_obj.result['race']['fair_metric_values']['ppv_parity'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)
                           
def test_compute_positive_predictive_ratio(ratio_metrics_setup):
    pred_underwriting_obj_ratio = ratio_metrics_setup
    # test without feature_mask
    expected_without_mask = 0.985
    result_without_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['gender']['fair_metric_values']['ppv_ratio'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = 1.017
    result_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['race']['fair_metric_values']['ppv_ratio'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)

def test_compute_auc_parity():
    mask_sk = pred_underwriting_obj.feature_mask['race'].reshape(-1)
    mask_p_sk = np.where(mask_sk ==1 , True, False)
    y_true_sk = pred_underwriting_obj.model_params[0].y_true.reshape(-1)[mask_p_sk]
    y_pred_sk = pred_underwriting_obj.model_params[0].y_pred.reshape(-1)[mask_p_sk]
    y_prob_sk = pred_underwriting_obj.model_params[0].y_prob.reshape(-1)[mask_p_sk]
    auc_p_sk = roc_auc_score(y_true_sk,y_prob_sk)

    mask_u_sk = np.where(mask_sk ==0 , True, False)
    y_true_sk_u = pred_underwriting_obj.model_params[0].y_true.reshape(-1)[mask_u_sk]
    y_pred_sk_u = pred_underwriting_obj.model_params[0].y_pred.reshape(-1)[mask_u_sk]
    y_prob_sk_u = pred_underwriting_obj.model_params[0].y_prob.reshape(-1)[mask_u_sk]
    auc_u_sk = roc_auc_score(y_true_sk_u,y_prob_sk_u)

    # test without feature_mask
    expected_without_mask = 0.008068999999999993    
    result_without_mask = pred_underwriting_obj.fair_metric_obj.result['gender']['fair_metric_values']['auc_parity'][0]    
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)
    # test feature_mask
    expected_without_mask_race = auc_p_sk-auc_u_sk
    result_mask = pred_underwriting_obj.fair_metric_obj.result['race']['fair_metric_values']['auc_parity'][0]
    assert round(result_mask, 3) == round(expected_without_mask_race, 3)

def test_compute_auc_ratio(ratio_metrics_setup):
    pred_underwriting_obj_ratio = ratio_metrics_setup
    # test without feature_mask
    expected_without_mask = 0.992
    result_without_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['gender']['fair_metric_values']['auc_ratio'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = 1.005
    result_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['race']['fair_metric_values']['auc_ratio'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)

def test_compute_log_loss_parity():
    # test without feature_mask
    expected_without_mask = 0.020
    result_without_mask = pred_underwriting_obj.fair_metric_obj.result['gender']['fair_metric_values']['log_loss_parity'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    mask_sk = pred_underwriting_obj.feature_mask['race'].reshape(-1)

    mask_p_sk = np.where(mask_sk ==1 , True, False)
    y_true_sk = pred_underwriting_obj.model_params[0].y_true.reshape(-1)[mask_p_sk]
    y_pred_sk = pred_underwriting_obj.model_params[0].y_pred.reshape(-1)[mask_p_sk]
    y_prob_sk = pred_underwriting_obj.model_params[0].y_prob.reshape(-1)[mask_p_sk]
    auc_p_sk = log_loss(y_true_sk,y_prob_sk)

    mask_u_sk = np.where(mask_sk ==0 , True, False)
    y_true_sk_u = pred_underwriting_obj.model_params[0].y_true.reshape(-1)[mask_u_sk]
    y_pred_sk_u = pred_underwriting_obj.model_params[0].y_pred.reshape(-1)[mask_u_sk]
    y_prob_sk_u = pred_underwriting_obj.model_params[0].y_prob.reshape(-1)[mask_u_sk]
    auc_u_sk = log_loss(y_true_sk_u,y_prob_sk_u)

    # test feature_mask
    expected_without_mask_race = auc_p_sk-auc_u_sk
    result_mask = pred_underwriting_obj.fair_metric_obj.result['race']['fair_metric_values']['log_loss_parity'][0]
    assert round(result_mask, 3) == round(expected_without_mask_race, 3)

def test_compute_log_loss_ratio(ratio_metrics_setup):
    pred_underwriting_obj_ratio = ratio_metrics_setup
    # test without feature_mask
    expected_without_mask = 0.854
    result_without_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['gender']['fair_metric_values']['log_loss_ratio'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = 0.867
    result_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['race']['fair_metric_values']['log_loss_ratio'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)
    
def test_consistency_score():
    # test for individual fairness, consistency score
    expected = 0.8787459459459459
    result = pred_underwriting_obj.fair_metric_obj.result['indiv_fair']['consistency_score']
    assert round(result, 3) == round(expected, 3)
    assert result >= 0 and result <= 1

def test_disable_individual_fairness():
    pred_underwriting_obj = PredictiveUnderwriting(model_params = [container_puw], fair_threshold = 80, fair_concern = "inclusive", \
                                                fair_priority = "benefit", fair_impact = "normal", fair_metric_type='difference', \
                                                tran_index=[1,2,3], tran_max_sample = 10, tran_max_display = 10, \
                                                tran_pdp_feature = ['age','payout_amount'])
    pred_underwriting_obj.evaluate(disable=['individual_fair'])
    assert pred_underwriting_obj.fair_metric_obj.result['indiv_fair'] is None

def test_multi_class_difference(multi_class_setup):
    container_clf = multi_class_setup
    clf_obj= BaseClassification(model_params = [container_clf], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal",fair_metric_type='difference', \
                            perf_metric_name = "accuracy", tran_index=[12,42], tran_max_sample=10, \
                            tran_pdp_feature = ['income','age'], tran_pdp_target='TR')
    clf_obj.evaluate(output=False)
    # Check result dict is not empty
    assert bool(clf_obj.fair_metric_obj.result)

    expected = -0.12032418952618457
    result = clf_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['demographic_parity'][0]
    assert round(result, 3) == round(expected, 3)

    expected = 0.07015762156225885
    result = clf_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['auc_parity'][0]
    assert round(result, 3) == round(expected, 3)

    expected = -0.426590059704548
    result = clf_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['log_loss_parity'][0]
    assert round(result, 3) == round(expected, 3)

def test_multi_class_ratio(multi_class_setup):
    container_clf = multi_class_setup
    clf_obj= BaseClassification(model_params = [container_clf], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal",fair_metric_type='ratio', \
                            perf_metric_name = "accuracy", tran_index=[12,42], tran_max_sample=10, \
                            tran_pdp_feature = ['income','age'], tran_pdp_target='TR')
    clf_obj.evaluate(output=False)
    # Check result dict is not empty
    assert bool(clf_obj.fair_metric_obj.result)

    expected = 1.4812967581047383
    result = clf_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['disparate_impact'][0]
    assert round(result, 3) == round(expected, 3)

    expected = 0.979360431051702
    result = clf_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['auc_ratio'][0]
    assert round(result, 3) == round(expected, 3)

    expected = 1.246610111046669
    result = clf_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['log_loss_ratio'][0]
    assert round(result, 3) == round(expected, 3)