import pickle
import sys
sys.path.append("C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit")
sys.path.append("C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit\\veritastool\\examples\\customer_marketing_example")
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.credit_scoring import CreditScoring
from veritastool.usecases.base_classification import BaseClassification
from veritastool.usecases.base_regression import BaseRegression
from veritastool.usecases.customer_marketing import CustomerMarketing
from veritastool.principles.transparency import Transparency
#from veritastool.model.modelwrapper import ModelWrapper
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from veritastool.util.errors import MyError

#creating a credit scoring use case
file = "C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit\\veritastool\\examples\\data\\credit_score_dict.pickle"
input_file = open(file, "rb")
cs = pickle.load(input_file)
cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
y_true = np.array(cs["y_test"])
y_pred = np.array(cs["y_pred"])
y_train = np.array(cs["y_train"])
p_grp = {'SEX': [[1]], 'MARRIAGE':[[1]]}
up_grp = {'SEX': [[2]], 'MARRIAGE':[[2]]}
x_train = cs["X_train"]
x_test = cs["X_test"]
model_name = "credit scoring"
model_type = "classification"
y_prob = cs["y_prob"]
model_object = LogisticRegression(C=0.1)
model_object.fit(x_train, y_train)
container = ModelContainer(y_true, p_grp, model_type, model_name,  y_pred, y_prob, y_train, x_train=x_train, \
                        x_test=x_test, model_object=model_object, up_grp=up_grp)
cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)

#creating a customer marketing use case
file_prop = "C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit\\veritastool\\examples\\data\\mktg_uplift_acq_dict.pickle"
file_rej = "C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit\\veritastool\\examples\\data\\mktg_uplift_rej_dict.pickle"
input_prop = open(file_prop, "rb")
input_rej = open(file_rej, "rb")
cm_prop = pickle.load(input_prop)
cm_rej = pickle.load(input_rej)
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
model_object_rej.fit(x_train_rej,y_train_rej)
model_object_prop.fit(x_train_rej,y_train_prop)
container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_pred_rej, y_prob = y_prob_rej, y_train= y_train_rej, \
                               p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, \
                               model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,\
                               pos_label=['TR', 'CR'], neg_label=['TN', 'CN'])
container_prop = container_rej.clone(y_true = y_true_prop, y_pred = y_pred_prop, y_prob = y_prob_prop, y_train= y_train_prop,\
                                model_object = model_object_prop,  pos_label=['TR', 'CR'], neg_label=['TN', 'CN'])
cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 80, \
                                  fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", \
                                  perf_metric_name = "expected_profit", fair_metric_name="auto", revenue = PROFIT_RESPOND, \
                                  treatment_cost =COST_TREATMENT, tran_index=[20,40], tran_max_sample=1000, \
                                  tran_pdp_feature= ['age','income'], tran_pdp_target='CR', tran_max_display = 6) 

def test_check_tran_index():
   
    msg = '[type_error]: tran_index: given <class \'int\'>, expected list at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=2, tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: tran_index: given a, expected List of integer at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=['a',2], tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: tran_index: given 50000, expected Index within range 1 - 22500 at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[50000], tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: tran_index: given -500, expected Index within range 1 - 22500 at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[1,2,-500], tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

def test_check_tran_max_sample():

    msg = '[value_error]: tran_max_sample: given 50000, expected Value between range 1 - 22500 at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 50000, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[type_error]: tran_max_sample: given <class \'list\'>, expected int or float at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = [25], tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: tran_max_sample: given 1000.56, expected Float value between 0 and 1 at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000.56, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

def test_check_tran_pdp_feature():
    msg = '[type_error]: tran_pdp_feature: given <class \'str\'>, expected list at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = 'abc', tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: tran_pdp_feature: given LimitBal, expected Feature value within available feature list at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ['LimitBal'], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: tran_pdp_feature: given a, expected Feature value within available feature list at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ["LIMIT_BAL","a","b","c"], tran_max_display = 10)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

def test_check_tran_pdp_target():
    #incorrect pdp target for a binary class model will give no error
    try:
        with pytest.raises(Exception) as toolkit_exit:
            cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_pdp_target = 5, tran_max_display = 10)
    except:
        assert True

    #creating a multi-class use case
    file_prop = "C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit\\veritastool\\examples\\data\\mktg_uplift_acq_dict.pickle"
    input_prop = open(file_prop, "rb")
    cm_prop = pickle.load(input_prop)
    y_true = cm_prop["y_test"]
    y_train = cm_prop["y_train"]
    model_obj = cm_prop['model']
    model_name = "base_classification" 
    model_type = "classification"
    y_prob = pd.DataFrame(cm_prop["y_prob"], columns=['CN', 'CR', 'TN', 'TR'])
    p_grp = {'isforeign':[[0]], 'isfemale':[[0]],'isforeign-isfemale':'maj_rest'}
    x_train = cm_prop["X_train"].drop(['ID'], axis = 1)
    x_test = cm_prop["X_test"].drop(['ID'], axis = 1)
    clf = cm_prop['model']
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    clf_container = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train, \
                            x_train=x_train, x_test=x_test, model_object=clf, \
                            pos_label=['TR','CR'], neg_label=['TN','CN'] ) 
    
    #incorrect pdp target for a multi class model
    msg = '[value_error]: tran_pdp_target: given 5, expected Target value from model class labels - [\'CN\' \'CR\' \'TN\' \'TR\'] at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        clf_obj= BaseClassification(model_params = [clf_container], fair_threshold = 80, fair_concern = "eligible", \
                                fair_priority = "benefit", fair_impact = "normal",fair_metric_type='difference', \
                                perf_metric_name = "accuracy", tran_index=[12,42], tran_max_sample=1000, \
                                tran_pdp_feature = ['income','age'], tran_pdp_target=5)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[type_error]: tran_pdp_target: given <class \'list\'>, expected str/int at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        clf_obj= BaseClassification(model_params = [clf_container], fair_threshold = 80, fair_concern = "eligible", \
                                fair_priority = "benefit", fair_impact = "normal",fair_metric_type='difference', \
                                perf_metric_name = "accuracy", tran_index=[12,42], tran_max_sample=1000, \
                                tran_pdp_feature = ['income','age'], tran_pdp_target=[1])
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

def test_check_tran_max_display():
    msg = '[value_error]: tran_max_display: given -9, expected Value between range 2 - 23 at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ["LIMIT_BAL"], tran_max_display = -9)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[type_error]: tran_max_display: given <class \'str\'>, expected int at _tran_check_input()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ["LIMIT_BAL"], tran_max_display = 'top5')
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ["LIMIT_BAL"], tran_max_display = 50)
    assert cre_sco_obj.tran_max_display == 23


def test_data_sampling():    
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[20,40], tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 16)
    cre_sco_obj._data_sampling()
    assert np.any(cre_sco_obj.tran_processed_data.index+1 == 20)
    assert np.any(cre_sco_obj.tran_processed_data.index+1 == 40)

    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                            fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                            tran_max_sample = 2, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 16)
    cre_sco_obj._data_sampling()
    assert np.any(cre_sco_obj.model_params[0].y_train[cre_sco_obj.tran_processed_data.index] == 0)
    assert np.any(cre_sco_obj.model_params[0].y_train[cre_sco_obj.tran_processed_data.index] == 1)

def test_top_features():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 10)
    cre_sco_obj._data_prep()
    assert cre_sco_obj.tran_top_features[0][:10] == ['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT5','PAY_AMT1',
                                                    'PAY_AMT2','BILL_AMT3','BILL_AMT6','PAY_AMT4','PAY_AMT3']

def test_global():
    pass

def test_local():
    pass

def test_compute_partial_dependence():
    pass

def test_compute_permutation_importance():
    #testing for classification
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[20,40], tran_max_sample = 1000, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 8)
    cre_sco_obj._data_prep()
    cre_sco_obj._compute_permutation_importance()
    assert list(np.round_(np.array(cre_sco_obj.permutation_importance['score']),6)) == [0.064667, 0.029333, 0.027733, 0.009733, 0.008133, 0.007333,
                                                                                        0.006133, 0.0056, 0.005467]
    assert list(cre_sco_obj.permutation_importance['features']) == ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT5', 'BILL_AMT3', 'PAY_AMT1', 'BILL_AMT6', 
                                                                    'PAY_AMT2', 'PAY_AMT4', 'LIMIT_BAL']


    #testing for regression
    file = "C:\\Sagar\\FS\\MAS\\Repo\\veritas-toolkit\\veritastool\\examples\\data\\regression_dict.pickle" 
    input_file = open(file, "rb")
    br = pickle.load(input_file)
    x_train = br["x_train"]
    x_test = br["x_test"]
    y_train = np.array(br["y_train"])
    y_true = np.array(br["y_test"])
    y_pred = np.array(br["y_pred"])
    p_grp = {'sex': [[1]], 'children': 'maj_min'}
    model_object = LinearRegression()
    model_name = "base_regression"
    model_type = "regression"
    model_object.fit(x_train,y_train)
    reg_container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_train=y_train, x_train=x_train, \
                           x_test=x_test, model_object=model_object)
    base_reg_obj= BaseRegression(model_params = [reg_container], fair_threshold = 80, perf_metric_name = "rmse", \
                             fair_concern = "eligible", fair_priority = "benefit", fair_impact = "normal", \
                             tran_index = [1,10,25], tran_max_sample = 1, tran_pdp_feature = ['age','bmi'])                      
    base_reg_obj._data_prep()
    base_reg_obj._compute_permutation_importance()
    
    assert list(np.round_(np.array(base_reg_obj.permutation_importance['score']),6)) == [9093.108145, 2173.331941, 823.94601, 125.229173, 50.07561, 48.964686]
    assert list(base_reg_obj.permutation_importance['features']) == ['smoker', 'age', 'bmi', 'children', 'sex', 'region']

    #testing for uplift model
    cm_uplift_obj._data_prep()
    cm_uplift_obj._compute_permutation_importance()
    
    assert list(np.round_(np.array(cm_uplift_obj.permutation_importance['score']),6)) == [9125.655707, 8533.327963, 3525.103062, 1494.305157, 1465.025095, 517.923479]
    assert list(cm_uplift_obj.permutation_importance['features']) == ['income', 'age', 'isforeign', 'noproducts', 'isfemale', 'didrespond']

def test_data_prep():
    pass

def test_explain():

    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                    fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                    tran_index=[20,40], tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 16)

    msg = '[value_error]: model_num: given 5, expected one of the following integers: [1] at explain()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj.explain(model_num=5)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[type_error]: model_num: given <class \'list\'>, expected int at explain()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj.explain(model_num=[1])
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[value_error]: local_index: given 50000, expected An integer value within the index range 1-22500 at explain()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj.explain(local_index=50000)
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    msg = '[type_error]: local_index: given <class \'list\'>, expected An integer value within the index range 1-22500 at explain()\n'
    with pytest.raises(Exception) as toolkit_exit:
        cre_sco_obj.explain(local_index=[10])
    assert toolkit_exit.type == MyError
    assert toolkit_exit.value.message == msg

    cre_sco_obj.explain(local_index=20)
    assert cre_sco_obj.tran_flag[0]['local'] == True
    assert cre_sco_obj.tran_flag[0]['total'] == False
    cre_sco_obj.explain()
    assert cre_sco_obj.tran_flag[0]['local'] == True
    assert cre_sco_obj.tran_flag[0]['total'] == True
    
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                        fair_priority = "benefit", fair_impact = "normal", perf_metric_name="accuracy", \
                        tran_index=[20,40], tran_max_sample = 100, tran_pdp_feature = ['LIMIT_BAL'], tran_max_display = 16)
    cre_sco_obj.explain()
    assert cre_sco_obj.tran_flag[0]['local'] == True
    assert cre_sco_obj.tran_flag[0]['total'] == True  

def test_tran_compile():
    pass