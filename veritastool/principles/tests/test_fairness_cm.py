import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.customer_marketing import CustomerMarketing
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.fairness import Fairness
import pytest
from veritastool.util.errors import *
#import selection, uplift, util
sys.path.append("veritas-toolkit/veritastool/examples/customer_marketing_example")
#Load Credit Scoring Test Data
#PATH = os.path.abspath(os.path.dirname(__file__)))
file_prop = "veritas-toolkit/veritastool/resources/data/mktg_uplift_acq_dict.pickle"
file_rej = "veritas-toolkit/veritastool/resources/data/mktg_uplift_rej_dict.pickle"
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
cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, \
                                  fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", \
                                  perf_metric_name = "expected_profit", fair_metric_name="auto", revenue = PROFIT_RESPOND, \
                                  treatment_cost =COST_TREATMENT, tran_index=[20,40], tran_max_sample=1000, \
                                  tran_pdp_feature= ['age','income'], tran_pdp_target='CR', tran_max_display = 6,fairness_metric_value_input = {'isforeign':{'rejected_harm': 0.2} })

#cm_uplift_obj.k = 1
cm_uplift_obj.compile()
# cm_uplift_obj.evaluate(visualize=True)
cm_uplift_obj.tradeoff()
cm_uplift_obj.feature_importance()
cm_uplift_obj.compile()
cm_uplift_obj.evaluate()

def test_evaluate():

    assert round(cm_uplift_obj.perf_metric_obj.result['perf_metric_values']['emp_lift'][0],3) == 0.171
   
def test_artifact():
    
    assert cm_uplift_obj.artifact['fairness']['features']['isforeign']['tradeoff']['th_x'].shape == cm_uplift_obj.artifact['fairness']['features']['isforeign']['tradeoff']['th_y'].shape
    assert cm_uplift_obj.artifact['fairness']['features']['isforeign']['tradeoff']['fair'].shape == cm_uplift_obj.artifact['fairness']['features']['isforeign']['tradeoff']['perf'].shape
    assert cm_uplift_obj.array_size == cm_uplift_obj.artifact['fairness']['perf_dynamic']['threshold'].shape[0]
    assert cm_uplift_obj.array_size == len(cm_uplift_obj.artifact['fairness']['perf_dynamic']['perf'])
    assert cm_uplift_obj.array_size == len(cm_uplift_obj.artifact['fairness']['perf_dynamic']['selection_rate'])
    
def test_fairness_conclusion():
    if cm_uplift_obj.fair_threshold < 1:
        assert cm_uplift_obj.fair_threshold == cm_uplift_obj.fair_conclusion['isforeign']['threshold']
    else:
        value = round((1 - cm_uplift_obj.fair_conclusion['isforeign']['threshold']) *100)
        assert cm_uplift_obj.fair_threshold == 85
    assert cm_uplift_obj.fair_conclusion['isforeign']['fairness_conclusion'] in ('fair','unfair')

def test_compute_fairness():
    if cm_uplift_obj.fairness_metric_value_input is not None :
        assert cm_uplift_obj.fairness_metric_value_input['isforeign']['rejected_harm'] == cm_uplift_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['rejected_harm'][0]
    
def test_fairness_metric_value_input_check():
    cm_uplift_obj.fairness_metric_value_input = {'INCOME': {'fpr_parity': 0.2}}
    cm_uplift_obj._fairness_metric_value_input_check()
    assert cm_uplift_obj.fairness_metric_value_input == None
    
    cm_uplift_obj.fairness_metric_value_input = {'isforeign': {'other_metric': 0.2}}
    cm_uplift_obj._fairness_metric_value_input_check()
    assert cm_uplift_obj.fairness_metric_value_input == None
    
def test_compile():

    assert cm_uplift_obj.evaluate_status == 1
    assert cm_uplift_obj.evaluate_status_cali == False
    assert cm_uplift_obj.evaluate_status_perf_dynamics == True
    assert cm_uplift_obj.tradeoff_status == 1
    assert cm_uplift_obj.feature_imp_status == 1
    assert cm_uplift_obj.feature_imp_status_loo == True
    assert cm_uplift_obj.feature_imp_status_corr == True
    
def test_compile_skip():
    cm_uplift_obj.feature_imp_status = 0
    cm_uplift_obj.tradeoff_status = 0
    cm_uplift_obj.feature_imp_status_corr = False
    #cm_uplift_obj.compile(skip_tradeoff_flag=1, skip_feature_imp_flag=1) unknown args
    assert cm_uplift_obj.feature_imp_status == 0 #-1
    assert cm_uplift_obj.tradeoff_status == 0 #-1
    
def test_tradeoff():

    assert round(cm_uplift_obj.tradeoff_obj.result['isforeign']['max_perf_point'][0],3) == 0.272
    cm_uplift_obj.model_params[0].y_prob = None
    cm_uplift_obj.tradeoff()
    assert cm_uplift_obj.tradeoff_status == -1
    cm_uplift_obj.tradeoff_obj.result= None
    cm_uplift_obj.tradeoff()
    assert cm_uplift_obj.tradeoff_status == -1
    
def test_feature_importance():
    cm_uplift_obj.feature_imp_status = 0
    cm_uplift_obj.evaluate_status = 0
    cm_uplift_obj.feature_importance()
    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][0],3) == 63332.82
    cm_uplift_obj.feature_imp_status = -1
    cm_uplift_obj.feature_importance()
    assert cm_uplift_obj.feature_imp_values == None
    x_train = np.array([1,2,3])
    x_test = np.array([1,2,3])
    cm_uplift_obj.feature_importance()
    assert isinstance(x_train, pd.DataFrame) == False
    assert isinstance(x_train, pd.DataFrame) == False
    
    
# def test_e_lift():
    # result = cm_uplift_obj._get_e_lift()
    # assert result == None

def test_feature_mask():
    assert len(cm_uplift_obj.model_params[0].x_test) == len(cm_uplift_obj.feature_mask['isforeign'])     

def test_base_input_check():
    cm_uplift_obj.fair_metric_name = 'mi_independence'
    cm_uplift_obj.fair_threshold = 43
    cm_uplift_obj.fairness_metric_value_input = {'isforeign': {'other_metric': 0.2}}
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._base_input_check()
    assert toolkit_exit.type == MyError
    
def test_model_type_input():
    cm_uplift_obj.model_params[0].model_type = 'svm'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
    cm_uplift_obj._model_type_to_metric_lookup[cm_uplift_obj.model_params[0].model_type] = ('uplift', 4, 4)
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._model_type_input()
    assert toolkit_exit.type == MyError

    cm_uplift_obj.model_params[0].model_type = 'uplift'
    cm_uplift_obj.model_params[1].model_name = 'duplicate'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
def test_fairness_tree():
    cm_uplift_obj.fair_impact = 'normal'
    #cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj._fairness_tree() == 'equal_opportunity'
    cm_uplift_obj.fair_concern = 'inclusive'
    #cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj._fairness_tree() == 'fpr_parity'
    cm_uplift_obj.fair_concern = 'both'
    #cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj._fairness_tree() == 'equal_odds'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'eligible'
    cm_uplift_obj.fair_priority = 'benefit'
    #cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj._fairness_tree() == 'ppv_parity'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'benefit'
    #cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj._fairness_tree() == 'fdr_parity'
    cm_uplift_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._fairness_tree()
    assert toolkit_exit.type == MyError
    cm_uplift_obj.fair_impact = 'normal'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'harm'
    #cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj._fairness_tree() == 'fpr_parity'
    
    cm_uplift_obj.fair_concern = 'eligible'
    cm_uplift_obj.fair_priority = 'benefit'
    cm_uplift_obj.fair_impact = 'normal'
    #cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj._fairness_tree(is_pos_label_favourable = False) == 'tnr_parity'
    cm_uplift_obj.fair_concern = 'inclusive'
    #cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj._fairness_tree(is_pos_label_favourable = False) == 'fnr_parity'
    cm_uplift_obj.fair_concern = 'both'
    #cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj._fairness_tree(is_pos_label_favourable = False) == 'neg_equal_odds'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'eligible'
    cm_uplift_obj.fair_priority = 'benefit'
    #cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj._fairness_tree(is_pos_label_favourable = False) == 'npv_parity'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'benefit'
    #cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj._fairness_tree(is_pos_label_favourable = False) == 'for_parity'
    cm_uplift_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert toolkit_exit.type == MyError
    cm_uplift_obj.fair_impact = 'normal'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'harm'
    #cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj._fairness_tree(is_pos_label_favourable = False) == 'fnr_parity'

def test_check_label():
    file_prop = "veritastool/examples/data/test_utility_data.pickle"
    input_prop = open(file_prop, "rb")
    cm_prop = pickle.load(input_prop)
    y_true_prop = cm_prop["y_true_prop"]
    y_true_new, pos_label2 = cm_uplift_obj._check_label(y=y_true_prop, pos_label=['TR', 'CR'], neg_label=['TN', 'CN'], obj_in=container_prop)
    labels, counts = np.unique(y_true_new, return_counts=True)
    assert np.array_equal(labels, np.array(['CN', 'CR', 'TN', 'TR']))
    assert np.array_equal(counts, np.array([3734, 2277, 2476, 1513]))
    
    y = np.array(['XR', 'CN', 'CR', 'TN', 'TR', 'XR', 'CN', 'CR', 'TN', 'TR'])
    msg = '[conflict_error]: pos_label, neg_label: inconsistent values [\'TR\', \'CR\', \'TN\', \'CN\'] at _check_label()\n'
    #catch the err poping out
    with pytest.raises(MyError) as toolkit_exit:
        y_new, pos_label2 = cm_uplift_obj._check_label(y=y, pos_label=['TR', 'CR'], neg_label=['TN', 'CN'], obj_in=container_prop)
    assert toolkit_exit.type == MyError
    # print(toolkit_exit.value.message)
    assert toolkit_exit.value.message == msg
