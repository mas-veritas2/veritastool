import pickle
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from veritastool.model.model_container import ModelContainer
from veritastool.model.modelwrapper import ModelWrapper
from veritastool.usecases.predictive_underwriting import PredictiveUnderwriting
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.transparency import Transparency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import imblearn
import shap
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *

#Load Predictive Underwriting Test Data
file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'underwriting_dict.pickle')
input_file = open(file, "rb")
puw = pickle.load(input_file)

#Model Contariner Parameters
y_true = np.array(puw["y_test"])
y_pred = np.array(puw["y_pred"])
y_train = np.array(puw["y_train"])
p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
up_grp = {'gender': [[0]], 'race': [[2, 3]] }
x_train = puw["X_train"]
x_test = puw["X_test"]
model_name = "pred_underwriting"
model_type = "classification"
y_prob = puw["y_prob"]

#Data Processing and Model Building
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

# Create Model Container
p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
up_grp = {'gender': [[0]], 'race': [[2, 3]] }
container = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                           x_test=x_test_final, model_object=model, up_grp=up_grp)

# Create Use Case Object
pred_underwriting_obj= PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "inclusive", \
                                        fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio',\
                                               tran_index=[1,2,3,20], tran_max_sample = 50, tran_max_display = 10, \
                                                      tran_pdp_feature = ['annual_premium','payout_amount'])

pred_underwriting_obj.compile()
pred_underwriting_obj.evaluate()
pred_underwriting_obj.tradeoff()
pred_underwriting_obj.feature_importance()
result = pred_underwriting_obj.perf_metric_obj.result, pred_underwriting_obj.fair_metric_obj.result

def test_evaluate():
    assert round(result[0]['perf_metric_values']['selection_rate'][0],3) == 0.816
   
def test_artifact():
    assert pred_underwriting_obj.artifact['fairness']['features']['gender']['tradeoff']['th_x'].shape == pred_underwriting_obj.artifact['fairness']['features']['gender']['tradeoff']['th_y'].shape
    assert pred_underwriting_obj.artifact['fairness']['features']['gender']['tradeoff']['fair'].shape == pred_underwriting_obj.artifact['fairness']['features']['gender']['tradeoff']['perf'].shape
    assert pred_underwriting_obj.array_size == pred_underwriting_obj.artifact['fairness']['perf_dynamic']['threshold'].shape[0]
    assert pred_underwriting_obj.array_size == len(pred_underwriting_obj.artifact['fairness']['perf_dynamic']['perf'])
    assert pred_underwriting_obj.array_size == len(pred_underwriting_obj.artifact['fairness']['perf_dynamic']['selection_rate'])
    
def test_fairness_conclusion():
    if pred_underwriting_obj.fair_threshold < 1:
        assert pred_underwriting_obj.fair_threshold == pred_underwriting_obj.fair_conclusion['gender']['threshold']
    else:
        value = round((1 - pred_underwriting_obj.fair_conclusion['gender']['threshold']) *100)
        assert pred_underwriting_obj.fair_threshold == value
    assert pred_underwriting_obj.fair_conclusion['gender']['fairness_conclusion'] in ('fair','unfair')

def test_compute_fairness():
    if pred_underwriting_obj.fairness_metric_value_input:
        assert pred_underwriting_obj.fairness_metric_value_input['gender']['fpr_parity'] == pred_underwriting_obj.fair_metric_obj.result['gender']['fair_metric_values']['fpr_parity'][0]
    
def test_fairness_metric_value_input_check():
    pred_underwriting_obj.fairness_metric_value_input = {'other_pvar': {'fpr_parity': 0.2}}
    pred_underwriting_obj._fairness_metric_value_input_check()
    assert pred_underwriting_obj.fairness_metric_value_input == None
    
    pred_underwriting_obj.fairness_metric_value_input = {'gender': {'other_metric': 0.2}}
    pred_underwriting_obj._fairness_metric_value_input_check()
    assert pred_underwriting_obj.fairness_metric_value_input == None
    
def test_compile():
    assert pred_underwriting_obj.evaluate_status == 1
    assert pred_underwriting_obj.evaluate_status_cali == True
    assert pred_underwriting_obj.evaluate_status_perf_dynamics == True
    assert pred_underwriting_obj.tradeoff_status == 1
    assert pred_underwriting_obj.feature_imp_status == 1
    assert pred_underwriting_obj.feature_imp_status_loo == True
    assert pred_underwriting_obj.feature_imp_status_corr == True
    
def test_compile_skip():
    pred_underwriting_obj.feature_imp_status = 0
    pred_underwriting_obj.tradeoff_status = 0
    pred_underwriting_obj.feature_imp_status_corr = False
    #pred_underwriting_obj.compile(skip_tradeoff_flag=1, skip_feature_imp_flag=1) unknown args
    assert pred_underwriting_obj.feature_imp_status == 0 #-1
    assert pred_underwriting_obj.tradeoff_status == 0 #-1
    
def test_tradeoff():
    assert round(pred_underwriting_obj.tradeoff_obj.result['gender']['max_perf_point'][0],3) == 0.510
    pred_underwriting_obj.model_params[0].y_prob = None
    pred_underwriting_obj.tradeoff()
    assert pred_underwriting_obj.tradeoff_status == -1
    pred_underwriting_obj.tradeoff_obj.result= None
    pred_underwriting_obj.tradeoff()
    assert pred_underwriting_obj.tradeoff_status == -1
    
def test_feature_importance():
    pred_underwriting_obj.feature_imp_status = 0
    pred_underwriting_obj.evaluate_status = 0
    pred_underwriting_obj.feature_importance()
    assert round(pred_underwriting_obj.feature_imp_values['gender']['gender'][0],3) == -0.029
    pred_underwriting_obj.feature_imp_status = -1
    pred_underwriting_obj.feature_importance()
    assert pred_underwriting_obj.feature_imp_values == None
    
def test_feature_importance_x_test_exception():   
    from veritastool.model.modelwrapper import ModelWrapper
    import numpy as np

    class xtestwrapper(ModelWrapper):
        """
        Abstract Base class to provide an interface that supports non-pythonic models.
        Serves as a template for users to define the

        """

        def __init__(self, model_obj, classes=[0, 1]):
            self.model_obj = model_obj
            self.classes_ = classes
            #self.output_file = output_file
           
        """
        Parameters
        ----------
        model_file : string
                Path to the model file. e.g. "/home/model.pkl"

        output_file : string
                Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        """

        def fit(self, X, y):
            
            
            """
            This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
            An example is as follows:
        
            train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
            import subprocess
            process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        """
            pass

        def predict(self, x_test, best_th = 0.43):
            pass
#             test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
#             test_preds = np.where(test_probs > best_th, 1, 0)
#             return test_preds

    model_object = None
    model_object = xtestwrapper(model_object)
    
    #Create Model Container 
    container = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train_resampled, x_train=x_train_final, \
                            x_test=x_test_final, model_object=model_object, up_grp=up_grp)

    #Create Use Case Object
    pred_underwriting_obj = PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                                                fair_priority = "benefit", fair_impact = "normal", fair_metric_name='auto', \
                                                tran_index=[1,2,3], tran_max_sample = 50, tran_max_display = 10, \
                                                tran_pdp_feature = ['age','payout_amount'])
    
    test = pred_underwriting_obj.feature_importance()
    assert test == None
    
def test_feature_importance_x_train_exception():   
    from veritastool.model.modelwrapper import ModelWrapper
    import numpy as np

    class xtrainwrapper(ModelWrapper):
        """
        Abstract Base class to provide an interface that supports non-pythonic models.
        Serves as a template for users to define the

        """

        def __init__(self, model_obj, classes=[0, 1]):
            self.model_obj = model_obj
            self.classes_ = classes
            #self.output_file = output_file
           
        """
        Parameters
        ----------
        model_file : string
                Path to the model file. e.g. "/home/model.pkl"

        output_file : string
                Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        """

        def fit(self, X, y):
            
            
            """
            This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
            An example is as follows:
        
            train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
            import subprocess
            process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        """
            pass

        def predict(self, x_test, best_th = 0.43):
            #error
            test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
            test_preds = np.where(test_probs > best_th, 1, 0)
            return test_preds
            
    model_object = None     
    model_object = xtrainwrapper(model_object)
    
    #Create Model Container 
    container = ModelContainer(y_true,  p_grp, model_type, model_name, y_pred, y_prob, y_train_resampled, x_train=x_train_final, \
                            x_test=x_test_final, model_object=model_object, up_grp=up_grp)

    #Create Use Case Object
    pred_underwriting_obj = PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "eligible", \
                                                fair_priority = "benefit", fair_impact = "normal", fair_metric_name='auto', \
                                                tran_index=[1,2,3], tran_max_sample = 50, tran_max_display = 10, \
                                                tran_pdp_feature = ['age','payout_amount'])

    test = pred_underwriting_obj.feature_importance()
    assert test == None
        
def test_e_lift():
    result = pred_underwriting_obj._get_e_lift()
    assert result == None
    
def test_feature_mask():
    assert len(pred_underwriting_obj.model_params[0].x_test) == len(pred_underwriting_obj.feature_mask['gender'])    
    
def test_base_input_check():
    pred_underwriting_obj.fair_metric_name = 'mi_independence'
    pred_underwriting_obj.fair_threshold = 43
    pred_underwriting_obj.fairness_metric_value_input = {'gender': {'other_metric': 0.2}}
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj._base_input_check()
    assert toolkit_exit.type == MyError
    
def test_model_type_input():
    pred_underwriting_obj.model_params[0].model_type = 'svm'
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
    pred_underwriting_obj._model_type_to_metric_lookup[pred_underwriting_obj.model_params[0].model_type] = ('classification', 2, 2)
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
    if pred_underwriting_obj.model_params[0].model_type == 'uplift' and pred_underwriting_obj.model_params[1].model_name != "clone" :
        pred_underwriting_obj.model_params[1].model_name = 'duplicate'
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
def test_fairness_tree():
    pred_underwriting_obj.fair_impact = 'normal'
    #pred_underwriting_obj._fairness_tree()
    assert pred_underwriting_obj._fairness_tree() == 'equal_opportunity'
    pred_underwriting_obj.fair_concern = 'inclusive'
    #pred_underwriting_obj._fairness_tree()
    assert pred_underwriting_obj._fairness_tree() == 'fpr_parity'
    pred_underwriting_obj.fair_concern = 'both'
    #pred_underwriting_obj._fairness_tree()
    assert pred_underwriting_obj._fairness_tree() == 'equal_odds_parity'
    pred_underwriting_obj.fair_impact = 'selective'
    pred_underwriting_obj.fair_concern = 'eligible'
    pred_underwriting_obj.fair_priority = 'benefit'
    #pred_underwriting_obj._fairness_tree()
    assert pred_underwriting_obj._fairness_tree() == 'ppv_parity'
    pred_underwriting_obj.fair_impact = 'selective'
    pred_underwriting_obj.fair_concern = 'inclusive'
    pred_underwriting_obj.fair_priority = 'benefit'
    #pred_underwriting_obj._fairness_tree()
    assert pred_underwriting_obj._fairness_tree() == 'fdr_parity'
    pred_underwriting_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj._fairness_tree()
    assert toolkit_exit.type == MyError
    pred_underwriting_obj.fair_impact = 'normal'
    pred_underwriting_obj.fair_concern = 'inclusive'
    pred_underwriting_obj.fair_priority = 'harm'
    #pred_underwriting_obj._fairness_tree()
    assert pred_underwriting_obj._fairness_tree() == 'fpr_parity'
    
    pred_underwriting_obj.fair_concern = 'eligible'
    pred_underwriting_obj.fair_priority = 'benefit'
    pred_underwriting_obj.fair_impact = 'normal'
    #pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False) == 'equal_opportunity'
    pred_underwriting_obj.fair_concern = 'inclusive'
    #pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False) == 'fpr_parity'
    pred_underwriting_obj.fair_concern = 'both'
    #pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False) == 'equal_odds'
    pred_underwriting_obj.fair_impact = 'selective'
    pred_underwriting_obj.fair_concern = 'eligible'
    pred_underwriting_obj.fair_priority = 'benefit'
    #pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False) == 'ppv_parity'
    pred_underwriting_obj.fair_impact = 'selective'
    pred_underwriting_obj.fair_concern = 'inclusive'
    pred_underwriting_obj.fair_priority = 'benefit'
    #pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False) == 'fdr_parity'
    pred_underwriting_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert toolkit_exit.type == MyError
    pred_underwriting_obj.fair_impact = 'normal'
    pred_underwriting_obj.fair_concern = 'inclusive'
    pred_underwriting_obj.fair_priority = 'harm'
    #pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False)
    assert pred_underwriting_obj._fairness_tree(is_pos_label_favourable = False) == 'fpr_parity'

def test_check_label():
    y = np.array([1,1,1,1,1,1,1])
    msg = '[value_error]: pos_label: given [1], expected not all y_true labels at _check_label()\n'
    #catch the err poping out
    with pytest.raises(MyError) as toolkit_exit:
        y_new, pos_label2 = pred_underwriting_obj._check_label(y=y, pos_label=[1], neg_label=[0])
    assert toolkit_exit.type == MyError
    # # print('====== test_check_label() =======\n')
    # # print(toolkit_exit.value.message)
    # # print('====== test_check_label() expected msg =======\n')
    # # print(msg)
    assert toolkit_exit.value.message == msg
    
    y = np.array([0,0,0,0,0,0])
    msg = '[value_error]: pos_label: given [1], expected {0} at _check_label()\n'
    #catch the err poping out
    with pytest.raises(MyError) as toolkit_exit:
        y_new, pos_label2 = pred_underwriting_obj._check_label(y=y, pos_label=[1], neg_label=[0])
    assert toolkit_exit.type == MyError
    # # print('====== test_check_label() =======\n')
    # # print(toolkit_exit.value.message)
    # # print('====== test_check_label() expected msg =======\n')
    # # print(msg)
    assert toolkit_exit.value.message == msg

def test_rootcause_group_difference():
    SEED=123
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "..", "..", "resources", "scenario_test", "shap_values_no_mask.npy")
    path = os.path.normpath(path)
    x_train_sample = x_train.sample(n=1000, random_state=SEED)
    shap_values = np.load(path)
    
    # Test case without feature_mask
    group_mask = np.where(x_train_sample.gender == 1, True, False)
    result = pred_underwriting_obj._rootcause_group_difference(shap_values, group_mask, x_train_sample.columns)
    expected = ['policy_duration', 'num_sp_policies', 'annual_premium', 'payout_amount',
                 'gender', 'num_life_policies', 'tenure', 'age', 'new_pol_last_3_years', 'num_pa_policies']
    assert list(result.keys()) == expected
    
    # Test case with feature_mask
    prot_var_df = x_train_sample['race']
    privileged_grp = pred_underwriting_obj.model_params[0].p_grp.get('race')[0]
    unprivileged_grp = pred_underwriting_obj.model_params[0].up_grp.get('race')[0]
    feature_mask = np.where(prot_var_df.isin(privileged_grp), True, -1)
    feature_mask = np.where(prot_var_df.isin(unprivileged_grp), False, feature_mask)
    indices = np.where(np.isin(feature_mask, [0, 1]))
    shap_values = shap_values[indices]
    group_mask = feature_mask[np.where(feature_mask != -1)].astype(bool)
    result = pred_underwriting_obj._rootcause_group_difference(shap_values, group_mask, x_train_sample.columns)
    expected = ['annual_premium', 'payout_amount', 'number_exclusions', 'policy_duration', 'num_sp_policies', 'num_life_policies', 
                'latest_purchase_product_category', 'num_pa_policies', 'new_pol_last_3_years', 'BMI']
    assert list(result.keys()) == expected

def test_rootcause():
    # Check p_var parameter: default all p_var
    pred_underwriting_obj.rootcause(p_var=[])
    assert bool(pred_underwriting_obj.rootcause_values) == True
    assert len(pred_underwriting_obj.rootcause_values.keys()) == 2
    assert len(pred_underwriting_obj.rootcause_values['gender'].values()) == 10
    
    # Check p_var parameter for 1 p_var, input_parameter_filtering to remove 'other_pvar'
    pred_underwriting_obj.rootcause(p_var=['gender', 'other_pvar'])
    assert bool(pred_underwriting_obj.rootcause_values) == True
    assert len(pred_underwriting_obj.rootcause_values.keys()) == 1
    assert len(pred_underwriting_obj.rootcause_values['gender'].values()) == 10
    
    # Check invalid p_var input
    with pytest.raises(MyError) as toolkit_exit:
        pred_underwriting_obj.rootcause(p_var=123)
    assert toolkit_exit.type == MyError
    
    # Check multi_class_label parameter: 
    pred_underwriting_obj.rootcause(multi_class_target=0)
    assert pred_underwriting_obj.rootcause_label_index == -1

def test_feature_imp_corr(capfd):
    pred_underwriting_obj.feature_imp_status_corr = False
    pred_underwriting_obj.feature_importance()

    # Check _print_correlation_analysis
    captured = capfd.readouterr()
    assert "* Surrogate detected for gender: num_life_policies" in captured.out

    # Check correlation_threshold
    pred_underwriting_obj.feature_imp_status_corr = False
    pred_underwriting_obj.feature_importance(correlation_threshold=0.6)
    assert pred_underwriting_obj.feature_imp_status_corr == True
    assert pred_underwriting_obj.correlation_threshold == 0.6

    # Disable correlation analysis
    pred_underwriting_obj.feature_imp_status_corr = False
    pred_underwriting_obj.feature_importance(disable=['correlation'])
    captured = capfd.readouterr()
    assert "Correlation matrix skipped" in captured.out

def test_compute_correlation():
    # Check top 3 features
    assert len(pred_underwriting_obj.corr_top_3_features) <= 6
    # Check surrogate features
    assert len(pred_underwriting_obj.surrogate_features['gender']) == 1
    assert 'num_life_policies' in pred_underwriting_obj.surrogate_features['gender']

@pytest.mark.parametrize("p_var", [(['gender']), ([])])
def test_mitigate_correlate(p_var, mitigate_correlate_setup):
    if p_var:
        mitigated_gender, result_mitigate1, _, _ = mitigate_correlate_setup
        # Check shape of mitigated x_train, x_test
        assert mitigated_gender['correlate'][0].shape == x_train.shape
        assert mitigated_gender['correlate'][1].shape == x_test.shape

        # Check that mitigated fair metric value is closer to neutral coefficient
        fair_metric_mitigated = result_mitigate1.get('gender')['fair_metric_values']['fpr_ratio'][0]
        assert abs(fair_metric_mitigated - 1) != abs(4.677 - 1)
    else:
        _, _, mitigated_all_pvars, result_mitigate2 = mitigate_correlate_setup
        # Check shape of mitigated x_train, x_test
        assert mitigated_all_pvars['correlate'][0].shape == x_train.shape
        assert mitigated_all_pvars['correlate'][1].shape == x_test.shape

        # Check that mitigated fair metric value is closer to neutral coefficient
        fair_metric_mitigated_gender = result_mitigate2.get('gender')['fair_metric_values']['fpr_ratio'][0]
        fair_metric_mitigated_race = result_mitigate2.get('race')['fair_metric_values']['fpr_ratio'][0]
        fair_metric_mitigated_intersect = result_mitigate2.get('gender-race-nationality')['fair_metric_values']['fpr_ratio'][0]
        assert abs(fair_metric_mitigated_gender - 1) != abs(4.677 - 1)
        assert abs(fair_metric_mitigated_race - 1) != abs(0.188 - 1)
        assert abs(fair_metric_mitigated_intersect - 1) != abs(3.930 - 1)

@pytest.mark.parametrize("p_var", [(['gender']), ([])])
def test_mitigate_threshold(p_var):
    pred_underwriting_obj.tradeoff()
    mitigated = pred_underwriting_obj.mitigate(p_var=p_var, method=['threshold'])
    assert mitigated['threshold'][0].shape == y_pred.shape
    if p_var:
        assert len(mitigated['threshold']) == 1
    else:
        assert all([
            mitigated['threshold'][1].shape == y_pred.shape,
            len(mitigated['threshold']) == 2,
        ])

def test_mitigate_threshold_mitigated(mitigate_threshold_setup):
    _, pred_underwriting_obj_mitg = mitigate_threshold_setup
    pred_underwriting_obj_mitg.evaluate(output=False)
    result_mitigate = pred_underwriting_obj_mitg.fair_metric_obj.result

    # Check that mitigated fair metric value is closer to neutral coefficient
    fair_metric_mitigated = result_mitigate.get('gender')['fair_metric_values']['fpr_ratio'][0]
    assert abs(fair_metric_mitigated - 1) != abs(4.677 - 1)

class MitigateWrapper(ModelWrapper):
    def __init__(self, model_obj, th, classes=[0, 1]):
        self.model_obj = model_obj
        self.classes_ = classes        
        self.th = th

    def fit(self, X, y):
        self.model_obj.fit(X, y)

    def predict(self, x_test):
        test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
        # Using bias mitigation thresholds
        test_preds = np.where(test_probs > self.th, 1, 0)
        return test_preds
    
    def predict_proba(self, x_test):
        return self.model_obj.predict_proba(x_test)
    
def get_row_threshold(X, column, groups, thresholds):
    th = np.zeros(len(X), dtype=float)
    for g, t in zip(groups, thresholds):
        group_mask = X[column] == g
        th[group_mask] = t
    return th

@pytest.fixture
def mitigate_threshold_setup():
    th = get_row_threshold(x_test, "gender", [1, 0], [0.422, 0.699])
    rfc_untrained = RandomForestClassifier(random_state=SEED)
    model_obj = MitigateWrapper(rfc_untrained, th)
    model_obj.fit(x_train_final, y_train_resampled)
    mitg_y_pred = model_obj.predict(x_test_final)
    mitg_y_prob = model_obj.predict_proba(x_test_final)[:, 1]
    container_mitg = ModelContainer(y_true, p_grp, model_type, model_name, mitg_y_pred, mitg_y_prob, y_train_resampled, x_train=x_train_final, \
                                     x_test=x_test_final, model_object=model_obj, up_grp=up_grp)
    pred_underwriting_obj_mitg = PredictiveUnderwriting(model_params=[container_mitg], fair_threshold=80, fair_concern="inclusive", \
                                                        fair_priority="benefit", fair_impact="normal", fair_metric_type='ratio', \
                                                        tran_index=[1,2,3], tran_max_sample=50, tran_max_display=10, \
                                                        tran_pdp_feature=['age','payout_amount'])
    yield container_mitg, pred_underwriting_obj_mitg

@pytest.fixture
def mitigate_correlate_setup():
    p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
    up_grp = {'gender': [[0]], 'race': [[2, 3]] }
    container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp)
    pred_underwriting_obj= PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "inclusive", \
                                            fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample = 50, tran_max_display = 10, \
                                                        tran_pdp_feature = ['annual_premium','payout_amount'])

    mitigated_gender = pred_underwriting_obj.mitigate(p_var=['gender'], method=['correlate'])
    x_train_mitigated = mitigated_gender['correlate'][0]
    x_test_mitigated = mitigated_gender['correlate'][1]
    y_pred_new = model.predict(x_test_mitigated)
    y_prob_new = model.predict_proba(x_test_mitigated)[:, 1]

    # Update Model Container
    p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
    up_grp = {'gender': [[0]], 'race': [[2, 3]]}
    container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred_new, y_prob_new, y_train, x_train=x_train_mitigated, \
                        x_test=x_test_mitigated, model_object=model, up_grp=up_grp)
    pred_underwriting_obj= PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "inclusive", \
                                        fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample = 50, tran_max_display = 10, \
                                                        tran_pdp_feature = ['annual_premium','payout_amount'])

    pred_underwriting_obj.evaluate(output=False)
    result_mitigate1 = pred_underwriting_obj.fair_metric_obj.result

    # Reinitialise Model Container
    p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
    up_grp = {'gender': [[0]], 'race': [[2, 3]]}
    container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp)
    pred_underwriting_obj= PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "inclusive", \
                                        fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample = 50, tran_max_display = 10, \
                                                        tran_pdp_feature = ['annual_premium','payout_amount'])

    mitigated_all_pvars = pred_underwriting_obj.mitigate(p_var=[], method=['correlate'])
    x_train_mitigated = mitigated_gender['correlate'][0]
    x_test_mitigated = mitigated_gender['correlate'][1]
    y_pred_new = model.predict(x_test_mitigated)
    y_prob_new = model.predict_proba(x_test_mitigated)[:, 1]

    # Update Model Container
    p_grp = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
    up_grp = {'gender': [[0]], 'race': [[2, 3]] }
    container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred_new, y_prob_new, y_train, x_train=x_train_mitigated, \
                        x_test=x_test_mitigated, model_object=model, up_grp=up_grp)
    pred_underwriting_obj= PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "inclusive", \
                                        fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample = 50, tran_max_display = 10, \
                                                        tran_pdp_feature = ['annual_premium','payout_amount'])

    pred_underwriting_obj.evaluate(output=False)
    result_mitigate2 = pred_underwriting_obj.fair_metric_obj.result
    yield mitigated_gender, result_mitigate1, mitigated_all_pvars, result_mitigate2

@pytest.mark.parametrize("p_var", [(['gender']), ([])])
def test_mitigate_reweigh(p_var):
    mitigated = pred_underwriting_obj.mitigate(p_var=p_var, method=['reweigh'], rw_weights=None, transform_x=None, transform_y=None)
    assert not pred_underwriting_obj.rw_is_transform
    assert mitigated['reweigh'][0].shape[0] == x_train.shape[0]
    assert isinstance(next(iter(mitigated['reweigh'][1].keys())), tuple)

    # Check values of sample weights computation based on ground truth
    if not p_var:
        assert round(mitigated['reweigh'][1][(0, 0, 1.0)], 3) == 0.827
    else:
        assert round(mitigated['reweigh'][1][(0, 1.0)], 3) == 0.947

@pytest.mark.parametrize("p_var", [(['gender']), ([])])
def test_mitigate_reweigh_categorical(p_var):
    x_train_rwg = x_train.copy()
    x_test_rwg = x_test.copy()
    x_train_rwg['gender'] = pd.Categorical.from_codes(x_train_rwg['gender'], categories=['male', 'female'])
    x_train_rwg['race'] = pd.Categorical.from_codes(x_train_rwg['race'], categories=['race_0', 'race_1', 'race_2', 'race_3', 'race_4'])
    x_test_rwg['gender'] = pd.Categorical.from_codes(x_test_rwg['gender'], categories=['male', 'female'])
    x_test_rwg['race'] = pd.Categorical.from_codes(x_test_rwg['race'], categories=['race_0', 'race_1', 'race_2', 'race_3', 'race_4'])
    p_grp = {'gender': [['male']], 'race': [['race_1']]}
    up_grp = {'gender': [['female']], 'race': [['race_2', 'race_3']] }

    container = ModelContainer(y_true, p_grp, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train_rwg, \
                            x_test=x_test_rwg, model_object=model, up_grp=up_grp)
    pred_underwriting_obj= PredictiveUnderwriting(model_params = [container], fair_threshold = 80, fair_concern = "inclusive", \
                                            fair_priority = "benefit", fair_impact = "normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample = 50, tran_max_display = 10, \
                                                        tran_pdp_feature = ['annual_premium','payout_amount'])
    
    mitigated = pred_underwriting_obj.mitigate(p_var=p_var, method=['reweigh'], rw_weights=None, transform_x=None, transform_y=None)
    assert not pred_underwriting_obj.rw_is_transform
    assert mitigated['reweigh'][0].shape[0] == x_train.shape[0]
    assert isinstance(next(iter(mitigated['reweigh'][1].keys())), tuple)

    # Check values of sample weights computation based on ground truth
    if not p_var:
        print(mitigated['reweigh'][1])
        assert round(mitigated['reweigh'][1][('male', 'race_0', 1.0)], 3) == 0.827
    else:
        assert round(mitigated['reweigh'][1][('male', 1.0)], 3) == 0.947

@pytest.mark.parametrize("p_grp, up_grp", [
    ({'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'maj_min'}, {'gender': [[0]], 'race': [[2, 3]]}),
    ({'gender': [[1]], 'race': 'maj_min'}, {'gender': [[0]]})
])
def test_policy_maj_min(p_grp, up_grp):
    p_grp_policy = p_grp
    up_grp_policy = up_grp
    container = ModelContainer(y_true, p_grp_policy, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp_policy)
    pred_underwriting_obj= PredictiveUnderwriting(model_params=[container], fair_threshold=80, fair_concern="inclusive", \
                                            fair_priority="benefit", fair_impact="normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample=50, tran_max_display=10, \
                                                        tran_pdp_feature=['annual_premium','payout_amount'])
    if 'gender-race-nationality' in p_grp:
        assert pred_underwriting_obj.model_params[0].p_grp['gender-race-nationality'][0] == ['0_1_1']
        assert pred_underwriting_obj.model_params[0].up_grp['gender-race-nationality'][0] == ['1_1_1']
    else:
        assert pred_underwriting_obj.model_params[0].p_grp['race'][0] == [1]
        assert pred_underwriting_obj.model_params[0].up_grp['race'][0] == [2]

@pytest.mark.parametrize("p_grp, up_grp", [
    ({'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'maj_rest'}, {'gender': [[0]], 'race': [[2, 3]]}),
    ({'gender': [[1]], 'race': 'maj_rest'}, {'gender': [[0]]})
])
def test_policy_maj_rest(p_grp, up_grp):
    p_grp_policy = p_grp
    up_grp_policy = up_grp
    container = ModelContainer(y_true, p_grp_policy, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp_policy)
    pred_underwriting_obj= PredictiveUnderwriting(model_params=[container], fair_threshold=80, fair_concern="inclusive", \
                                            fair_priority="benefit", fair_impact="normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample=50, tran_max_display=10, \
                                                        tran_pdp_feature=['annual_premium','payout_amount'])
    if 'gender-race-nationality' in p_grp:
        expected_upgrp = ['1_1_1', '0_1_0', '0_3_1', '1_3_1', '0_2_1', '1_2_0', '1_2_1', '1_1_0', '1_4_1', '0_2_0', '0_4_1', '0_1_2', '1_4_0', '0_4_0', '0_3_0', '1_0_1', '1_1_2', '0_2_2', '1_3_0']
        assert pred_underwriting_obj.model_params[0].p_grp['gender-race-nationality'][0] == ['0_1_1']
        assert set(pred_underwriting_obj.model_params[0].up_grp['gender-race-nationality'][0]) == set(expected_upgrp)
    else:
        expected_upgrp = [3, 2, 4, 0]
        assert pred_underwriting_obj.model_params[0].p_grp['race'][0] == [1]
        assert set(pred_underwriting_obj.model_params[0].up_grp['race'][0]) == set(expected_upgrp)

@pytest.mark.parametrize("p_grp, up_grp", [
    ({'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}, {'gender': [[0]], 'race': [[2, 3]]}),
    ({'gender': [[1]], 'race': 'max_bias'}, {'gender': [[0]]})
])
def test_policy_max_bias(p_grp, up_grp):
    p_grp_policy = p_grp
    up_grp_policy = up_grp
    container = ModelContainer(y_true, p_grp_policy, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp_policy)
    pred_underwriting_obj= PredictiveUnderwriting(model_params=[container], fair_threshold=80, fair_concern="inclusive", \
                                            fair_priority="benefit", fair_impact="normal", fair_metric_type='ratio',\
                                                tran_index=[1,2,3,20], tran_max_sample=50, tran_max_display=10, \
                                                        tran_pdp_feature=['annual_premium','payout_amount'])
    if 'gender-race-nationality' in p_grp:
        assert pred_underwriting_obj.model_params[0].p_grp['gender-race-nationality'][0] == ['1_1_1']
        assert pred_underwriting_obj.model_params[0].up_grp['gender-race-nationality'][0] == ['0_1_1']
    else:
        assert pred_underwriting_obj.model_params[0].p_grp['race'][0] == [2]
        assert pred_underwriting_obj.model_params[0].up_grp['race'][0] == [1]

@pytest.mark.parametrize("fair_metric_name, ", [
    ('log_loss_parity'),
    ('auc_parity')
])
def test_policy_max_bias_y_prob(fair_metric_name):
    # Check direction indicator anda y_prob-based metrics
    p_grp_policy = {'gender': [[1]], 'race': [[1]], 'gender-race-nationality':'max_bias'}
    up_grp_policy = {'gender': [[0]], 'race': [[2, 3]]}
    container = ModelContainer(y_true, p_grp_policy, model_type, model_name, y_pred, y_prob, y_train, x_train=x_train, \
                            x_test=x_test_final, model_object=model, up_grp=up_grp_policy)
    pred_underwriting_obj= PredictiveUnderwriting(model_params=[container], fair_threshold=80, fair_concern="inclusive", \
                                            fair_priority="benefit", fair_impact="normal", fair_metric_name=fair_metric_name,\
                                                tran_index=[1,2,3,20], tran_max_sample=50, tran_max_display=10, \
                                                        tran_pdp_feature=['annual_premium','payout_amount'])
    if fair_metric_name == 'auc_parity':
        assert pred_underwriting_obj.model_params[0].p_grp['gender-race-nationality'][0] == ['1_1_1']
        assert pred_underwriting_obj.model_params[0].up_grp['gender-race-nationality'][0] == ['0_1_1']
    elif fair_metric_name == 'log_loss_parity':
        assert pred_underwriting_obj.model_params[0].p_grp['gender-race-nationality'][0] == ['0_1_1']
        assert pred_underwriting_obj.model_params[0].up_grp['gender-race-nationality'][0] == ['1_1_1']