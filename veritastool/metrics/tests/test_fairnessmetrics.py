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
from sklearn.metrics import roc_auc_score, log_loss
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../veritastool/examples/customer_marketing_example'))
sys.path.append(module_path)
import selection, uplift, util

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

import pickle
import numpy as np
import pandas as pd
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.predictive_underwriting import PredictiveUnderwriting
from veritastool.metrics.fairness_metrics import FairnessMetrics
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import imblearn

#Load Predictive Underwriting Test Data
file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'underwriting_dict.pickle')
input_file = open(file, "rb")
puw = pickle.load(input_file)

#Model Contariner Parameters
y_true = np.array(puw["y_test"])
y_pred = np.array(puw["y_pred"])
y_train = np.array(puw["y_train"])
p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race':'max_bias'}
up_grp = {'gender': [[0]], 'race': [[2, 3]] }
x_train = puw["X_train"]
x_test = puw["X_test"]
y_prob = puw["y_prob"]

#Building Model
SEED=123
obj_cols = x_train.select_dtypes(include='object').columns
for col in obj_cols:
    le = LabelEncoder()
    le.fit(pd.concat([x_train[col], x_test[col]]))
    x_train[col] = le.transform(x_train[col])
    x_test[col] = le.transform(x_test[col])
under = RandomUnderSampler(sampling_strategy=0.5, random_state=SEED)
over = SMOTE(random_state=SEED)
pipe = imblearn.pipeline.Pipeline([
    ('under', under),
    ('over', over)
])
x_train_resampled, y_train_resampled = pipe.fit_resample(x_train, y_train)
x_train_final = pd.DataFrame(x_train_resampled, columns=x_train.columns)
x_test_final = x_test

model = puw["model"]
model_name = "pred_underwriting"
model_type = "classification"

#Create Model Container 
container_puw = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train_resampled, x_train=x_train_final, \
                           x_test=x_test_final, model_object=model, up_grp=up_grp)

#Create Use Case Object
pred_underwriting_obj = PredictiveUnderwriting(model_params = [container_puw], fair_threshold = 80, fair_concern = "inclusive", \
                                              fair_priority = "benefit", fair_impact = "normal", fair_metric_type='difference', \
                                              tran_index=[1,2,3], tran_max_sample = 10, tran_max_display = 10, \
                                              tran_pdp_feature = ['age','payout_amount'])

@pytest.fixture
def ratio_metrics_setup():
    p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race':'max_bias'}
    up_grp = {'gender': [[0]], 'race': [[2, 3]] }
    container_puw = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train_resampled, x_train=x_train_final, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp)
    pred_underwriting_obj = PredictiveUnderwriting(model_params = [container_puw], fair_threshold = 80, fair_concern = "inclusive", \
                                                fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio', \
                                                tran_index=[1,2,3], tran_max_sample = 10, tran_max_display = 10, \
                                                tran_pdp_feature = ['age','payout_amount'])
    pred_underwriting_obj.evaluate(output=False)
    
    yield pred_underwriting_obj

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

def test_translate_metric():
    cre_sco_obj.feature_importance()
    assert cre_sco_obj.feature_imp_values != None
    cm_uplift_obj.feature_importance()
    assert cm_uplift_obj.feature_imp_values != None

def test_compute_wape_parity():
    cre_sco_obj.curr_p_var = 'SEX'
    cre_sco_obj.y_true = [[],[]]
    cre_sco_obj.y_true[0] = np.array(y_true)
    cre_sco_obj.y_true[1] = np.array(y_true)
    cre_sco_obj.y_pred = [[],[]]
    cre_sco_obj.y_pred[0] = np.array(y_pred)
    cre_sco_obj.y_pred[1] = np.array(y_pred)
    result = FairnessMetrics._compute_wape_parity(cre_sco_obj)
    assert result == (0.216670530952933, 0.4008810572687225)
    
def test_compute_mape_parity():
    cre_sco_obj.sample_weight = [[],[]]
    cre_sco_obj.sample_weight[0] = np.array([0.7 for x in range(len(y_pred))])
    cre_sco_obj.sample_weight[1] = np.array([0.7 for x in range(len(y_pred))])
    result = FairnessMetrics._compute_mape_parity(cre_sco_obj)
    assert result == (46294276723477.69, 394254512833780.9)

def test_compute_mape_parity():
    cre_sco_obj.sample_weight = [[],[]]
    np.random.seed(0)

    cre_sco_obj.sample_weight[0] = np.random.random_sample((len(y_pred),))
    cre_sco_obj.sample_weight[1] = np.random.random_sample((len(y_pred),))
    result = FairnessMetrics._compute_mape_parity(cre_sco_obj)
    assert result == (50612541836597.81, 440356453891928.2)
    
def test__compute_rmse_parity():
    result = FairnessMetrics._compute_rmse_parity(cre_sco_obj)
    assert result == (0.16043449543124988, 0.5535316670230406)    

def test__compute_rmse_parity():
    result = FairnessMetrics._compute_rmse_parity(cre_sco_obj)
    assert result == (0.14359862866686257, 0.5529818274559376)

def test_compute_benefit_from_acquiring():
    cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", fair_metric_name = 'acquire_benefit', revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
    cm_uplift_obj.feature_importance()
    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][1],3) == -0.045

def test_compute_disparate_impact_rejection_inference():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", \
                           fair_priority = "benefit", fair_impact = "significant", \
                           fair_metric_name = 'disparate_impact', fair_metric_type='ratio', \
                           num_applicants =num_applicants,  base_default_rate=base_default_rate, \
                           tran_index=[20,40], tran_max_sample = 10, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10, \
                           )
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == -0.302
    
def test_compute_disparate_impact():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", \
                           fair_priority = "benefit", fair_impact = "significant", \
                           fair_metric_name = 'disparate_impact', fair_metric_type='ratio', \
                           tran_index=[20,40], tran_max_sample = 10, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == -0.283

def test_compute_positive_predictive_parity():
    # test without feature_mask
    expected_without_mask = 0.010916822290745
    result_without_mask = pred_underwriting_obj.fair_metric_obj.result['gender']['fair_metric_values']['ppv_parity'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = -0.0116318443921323
    result_mask = pred_underwriting_obj.fair_metric_obj.result['race']['fair_metric_values']['ppv_parity'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)
                           
def test_compute_positive_predictive_ratio(ratio_metrics_setup):
    pred_underwriting_obj_ratio = ratio_metrics_setup
    # test without feature_mask
    expected_without_mask = 0.988726936227686
    result_without_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['gender']['fair_metric_values']['ppv_ratio'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = 1.0120921133020016
    result_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['race']['fair_metric_values']['ppv_parity'][0]
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
    print(pred_underwriting_obj_ratio.fair_metric_obj.result)
    # test without feature_mask
    expected_without_mask = 0.9940139989449025
    result_without_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['gender']['fair_metric_values']['auc_ratio'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = 1.036983852285801
    result_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['race']['fair_metric_values']['auc_ratio'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)

def test_compute_log_loss_parity():
    # test without feature_mask
    expected_without_mask = 1.3168670002582914
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
    expected_without_mask = 0.8088605309684556
    result_without_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['gender']['fair_metric_values']['log_loss_ratio'][0]
    assert round(result_without_mask, 3) == round(expected_without_mask, 3)

    # test feature_mask
    expected_mask = 1.0016924334136152
    result_mask = pred_underwriting_obj_ratio.fair_metric_obj.result['race']['fair_metric_values']['log_loss_ratio'][0]
    assert round(result_mask, 3) == round(expected_mask, 3)
    
def test_consistency_score():
    # test for individual fairness, consistency score
    expected = 0.8787459459459459
    result = pred_underwriting_obj.fair_metric_obj.result['indiv_fair']['consistency_score']
    assert round(result, 3) == round(expected, 3)
    assert result >= 0 and result <= 1