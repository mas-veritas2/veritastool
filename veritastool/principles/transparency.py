import shap
import numpy as np 
import pandas as pd 
import base64
import io
import matplotlib.pyplot as pl
from sklearn.inspection import PartialDependenceDisplay
from shap import Explanation
from ..util.errors import *
from ..util.utility import *
from ..config.constants import Constants
from tqdm.auto import tqdm
from ..metrics.performance_metrics import PerformanceMetrics
import copy 
from ..util.utility import process_y_prob, check_data_unassigned
from IPython.display import display_html
from IPython.core.display import HTML

class Transparency:
    """
    Base Class with attributes for transparency analysis. 
    """
    def __init__(self, tran_index = [1], tran_max_sample = 1, tran_pdp_feature = [], tran_pdp_target=None, tran_max_display = 10,tran_features=[]):
        """
        Parameters
        ------------------
        tran_index : list
                It holds the list of indices given by the user for which the user wants to see the local interpretability.

        tran_list_of_features : str/list
                Stores the string or list of features given by the user. String input can either be "top n" or "all".
                If none is given, the top 10 features will be chosed while sampling the data.

        tran_max_records : float
                Stores the number or percentage of rows to be included for sampling. 
                For a number less than 1, it will be considered as percentage. Any number above 1 will be counted as number of rows.
                By default 100 records will be included. 

        tran_pdp_feature : list
                Stores the list of the top 2 features for which user wants to see the partial dependence plot. 
                If none is passed, plot for top 2 features will be shown by default. 

        Instance Attributes
        -------------------
        tran_results : dict, default=None
                Stores the dictionary containing the all the required transparency output.

        tran_shap_values : Explanation object, default=None
                Stores the shapley explanation values obtained based on the model and data passed.

        tran_processed_data : dataframe, default=None
                Stores the processed dataframe which will be used to create the shap values. 
                The data will be created based on the list of features and max records that the user gives.

        tran_status_local : Boolean, default=False
                Stores a flag for whether the local interpretability plot points for all indices given by the user has been executed or not.
                False: Plot points not executed for all indices.
                True: Plot points executed for all indices.

        tran_status_total : Boolean, default=False
                Stores a flag for whether all the transparency analysis functions have been executed or not.
                False: All results of transparency analysis are not computed.
                True: All results of transparency analysis are computed.
        
        """
        self.tran_max_sample = tran_max_sample
        self.tran_pdp_feature = tran_pdp_feature
        self.tran_pdp_target = tran_pdp_target
        self.tran_max_display = tran_max_display
        self.tran_index = tran_index
        self.tran_features = tran_features
        self.tran_shap_values = {} 
        self.tran_shap_extra = {} 
        self.tran_processed_data = None
        self.tran_top_features = {}
        self.tran_pdp_feature_list={}
        self.permutation_importance = pd.DataFrame(columns = ["feature", "diff", "neg_flag"])
        self.tran_flag = {'data_sampling_flag':False} 
        self.tran_results = {'permutation_score':'','model_list':[]}

        for i in range(len(self.model_params)):
            self.tran_results['model_list'].append({'id':i,
                                    'summary_plot':'',
                                    'local_interpretability':[],
                                    'partial_dependence_plot':{},
                                    'plot':{'local_plot':{},
                                            'class_index':{},
                                            'pdp_plot':{}}})
            self.tran_flag[i] = {'interpret':False,'partial_dep':False,'perm_imp':False,'data_prep_flag':False}

    def _tran_check_input(self):
        self._check_tran_index()
        self._check_tran_max_sample()
        self._check_tran_pdp_feature()
        self._check_tran_pdp_target()
        self._check_tran_max_display() 
        self._check_tran_features()
        self.err.pop()                                   

    def _check_tran_index(self):
        if(type(self.tran_index) not in [list,np.ndarray,pd.Series]):   
            self.err.push('type_error', var_name="tran_index", given=type(self.tran_index), expected='list or numpy array or pandas series', function_name="_tran_check_input")
        else:
            if(type(self.tran_index) in [np.ndarray,pd.Series]):
                self.tran_index = self.tran_index.tolist()
            for j,i in enumerate(self.tran_index):
                if(type(i) not in [int,float]):
                    self.err.push('type_error', var_name="tran_index", given=type(i), expected='integer values', function_name="_tran_check_input")
                elif(not(1<=int(i)<=self.model_params[0].x_train.shape[0])):
                    self.err.push('value_error', var_name="tran_index", given=i, expected='Index within range 1 - '+ str(self.model_params[0].x_train.shape[0]), function_name="_tran_check_input")    
                if(type(i)==float):
                    self.tran_index[j]=int(i)
                
    def _check_tran_max_sample(self):
        if(type(self.tran_max_sample) not in [int,float]):
            self.err.push('type_error', var_name="tran_max_sample", given=type(self.tran_max_sample), expected='int or float', function_name="_tran_check_input")
        elif((type(self.tran_max_sample)==float) and not(0<self.tran_max_sample<1)):
            self.err.push('value_error', var_name="tran_max_sample", given=self.tran_max_sample, expected='Float value between 0 and 1', function_name="_tran_check_input")
        elif((type(self.tran_max_sample)==int) and not(1<=self.tran_max_sample<=self.model_params[0].x_train.shape[0])):
            self.err.push('value_error', var_name="tran_max_sample", given=self.tran_max_sample, expected='Value between range 1 - ' + str(self.model_params[0].x_train.shape[0]), function_name="_tran_check_input")

    def _check_tran_pdp_feature(self):    
        if(type(self.tran_pdp_feature)!=list):
            self.err.push('type_error', var_name="tran_pdp_feature", given=type(self.tran_pdp_feature), expected='list', function_name="_tran_check_input")
        else:
            for i in self.tran_pdp_feature[:2]:
                if(type(i)!=str):
                    self.err.push('type_error', var_name="tran_pdp_feature", given=type(i), expected='list of string', function_name="_tran_check_input")    
                elif(i not in self.model_params[0].x_train.columns):
                    self.err.push('value_error', var_name="tran_pdp_feature", given=i, expected='Feature value within available feature list', function_name="_tran_check_input")

    def _check_tran_pdp_target(self):
        if(self.model_params[0].model_type!='regression'):
            if(self.tran_pdp_target is not None) and len(self.model_params[0].model_object.classes_)>2:
                if(type(self.tran_pdp_target) not in [str,int]): 
                    self.err.push('type_error', var_name="tran_pdp_target", given=type(self.tran_pdp_target), expected='str/int', function_name="_tran_check_input")
                elif(self.tran_pdp_target not in self.model_params[0].model_object.classes_):
                    self.err.push('value_error', var_name="tran_pdp_target", given=self.tran_pdp_target, expected='Target value from model class labels - ' + str(self.model_params[0].model_object.classes_), function_name="_tran_check_input")

    def _check_tran_max_display(self):
        if(type(self.tran_max_display)==float):
            self.tran_max_display = int(self.tran_max_display)        
        if(type(self.tran_max_display)!=int):
            self.err.push('type_error', var_name="tran_max_display", given=type(self.tran_max_display), expected='int', function_name="_tran_check_input")
        elif(self.tran_max_display == 0):
            self.tran_max_display = self.model_params[0].x_train.shape[1]
        elif(self.tran_max_display<2):
            self.err.push('value_error', var_name="tran_max_display", given=self.tran_max_display, expected='Value between range 2 - ' + str(self.model_params[0].x_train.shape[1]), function_name="_tran_check_input")
        else:
            self.tran_max_display = min(self.tran_max_display,self.model_params[0].x_train.shape[1])    
    
    def _check_tran_features(self):    
        if(type(self.tran_features)!=list):
            self.err.push('type_error', var_name="tran_features", given=type(self.tran_features), expected='list', function_name="_tran_check_input")
        else:
            for i in self.tran_features:
                if(type(i)!=str):
                    self.err.push('type_error', var_name="tran_features", given=type(self.tran_features), expected='str', function_name="_tran_check_input")    
                elif(i not in self.model_params[0].x_train.columns):
                    self.err.push('value_error', var_name="tran_features", given=i, expected='Feature value within available feature list', function_name="_tran_check_input")

    def _shap(self,model_num=0):
        """
        Calculates shap values for the given model and dataset 

        Returns
        This function does not return anything. It stores the shap values in the attribute tran_shap_values. 
        ----------
        """
        if(self.model_params[model_num].model_type == 'regression'):
            explainer_shap = shap.Explainer(self.model_params[model_num].model_object.predict,self.tran_processed_data) 
            explanation = explainer_shap(self.tran_processed_data)
            self.tran_shap_extra[model_num] = explanation.base_values
            self.tran_shap_values[model_num] = explanation.values
        else:
            explainer_shap = shap.Explainer(self.model_params[model_num].model_object.predict_proba,self.tran_processed_data) 
            explanation = explainer_shap(self.tran_processed_data)
            base = explanation.base_values
            shap_values= np.moveaxis(explanation.values, -1, 0) 
            if(shap_values.shape[0]==2):
                idx = list(self.model_params[0].model_object.classes_).index(self.model_params[0].pos_label[0])
                shap_values = shap_values[idx]
                base = base[:,idx]
            else:
                shap_values = list(shap_values)
            self.tran_shap_values[model_num] = shap_values
            self.tran_shap_extra[model_num] = base
        
    def _data_sampling(self):
        """
        Identifies the list of features basis the user's input to be used in processed data.

        Returns
        ----------
        This function does not return anything. It creates a list of features to be used for sampling.
                
        For customer mkting: list of list []
        """ 

        if 0 < self.tran_max_sample < 1:
            self.tran_max_sample = round(self.tran_max_sample*self.model_params[0].x_train.shape[0])
        elif 1 < self.tran_max_sample <= self.model_params[0].x_train.shape[0]:
            self.tran_max_sample = round(self.tran_max_sample)
        else:
            self.tran_max_sample = self.model_params[0].x_train.shape[0]

        self.tran_processed_data = self.model_params[0].x_train.reset_index(drop=True)
        self.tran_processed_data = self.tran_processed_data.sample(n=self.tran_max_sample,random_state=0)

        if(self.model_params[0].model_type!='regression'):
            y_train=pd.Series(self.model_params[0].y_train).reset_index(drop=True)
            label = set(y_train[self.tran_processed_data.index])
            missed = set(self.model_params[0].model_object.classes_) - label
            if(len(missed)>0):
                new_rows = []
                for j in zip(y_train.index, y_train.values):
                    if j[1] in missed:
                        new_rows += [j[0]]
                        if len(missed) == 1:
                            break
                        else:    
                            missed.remove(j[1])
                self.tran_processed_data = pd.concat([self.tran_processed_data,self.model_params[0].x_train.iloc[new_rows].set_index([new_rows])])

        diff = list(set([x - 1 for x in self.tran_index]) - set(self.tran_processed_data.index))
        self.tran_processed_data = pd.concat((self.tran_processed_data, self.model_params[0].x_train.iloc[diff].set_index([diff])), axis = 0)   

    def _top_features(self,model_num=0): 
        """
        Function to create tran_list_of_features and tran_pdp_feature basis the user input. The computation is done basis the first model obtained from the container incase more than 1 model is passed. Feature importance for the features is also calculated here for selection of top features basis the shap values. 
        """
        if(type(self.tran_shap_values[model_num])==list):
            importances = np.sum(np.mean(np.abs(self.tran_shap_values[model_num]),axis=1),axis=0) 
        else:
            importances = np.mean(np.abs(self.tran_shap_values[model_num]),axis=0)
        features = self.model_params[0].x_train.columns
        feature_imp = pd.DataFrame({'Feature_name':features, 'Mean|shap|': importances})
        feature_imp.sort_values(by=['Mean|shap|'], ascending=False, inplace=True)
        self.tran_top_features[model_num] = feature_imp
        
    def _plot_encode(self, f, plot = None):
        buff = io.BytesIO()
        f.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = f.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))  
        
        image_64_encodeutf = ''
        if(plot is not None):
            buffer = io.BytesIO()
            f.savefig(buffer, format = 'png')
            image_64_encode = base64.b64encode(buffer.getvalue())
            image_64_encodeutf=image_64_encode.decode('utf-8') 
        if(image_64_encodeutf!=''):
            return im,image_64_encodeutf
        else:
            return im
          
  
    def _global(self,model_num=0):
        """
        Computes the global interpretability on the processed data.

        Returns
        ----------
        This function does not return anything. It provides the encoded plot image in the results dictionary.
        
        tran_list_of_features will only for list/string
        
        """
        if(self.tran_features==[]):
            if(type(self.tran_shap_values[model_num])==list):
                f= pl.gcf()
                shap.summary_plot(self.tran_shap_values[model_num], self.tran_processed_data, class_names=self.model_params[model_num].model_object.classes_, max_display=self.tran_max_display, show= False)
                im,image_base_64 = self._plot_encode(f, plot="summary_plot")
                pl.close()
            else:
                f= pl.gcf()
                shap.summary_plot(self.tran_shap_values[model_num], self.tran_processed_data, max_display=self.tran_max_display, show= False)
                im,image_base_64 = self._plot_encode(f, plot="summary_plot")
                pl.close()
            
            self.tran_results['model_list'][model_num]['summary_plot'] = image_base_64
            self.tran_results['model_list'][model_num]['plot']['summary'] = im
        
        else:
            summary = self.tran_top_features[model_num][np.isin(self.tran_top_features[model_num], self.tran_features).any(axis=1)]
            self.tran_results['model_list'][model_num]['summary_plot'] = summary.to_dict(orient='records')
            self.tran_results['model_list'][model_num]['plot']['summary'] = ''

    def _local(self, n, model_num=0):
        """
        Computes the local interpretability values for the index provided by the user.

        Parameters
        -----------
        n : int
                Index for which the local interpretability is to be calculated.

        update_flag : True
                If True, the tran_results dictionary is updated. 
                If False, the dictionary is not updated. Only the plot is shown in the console
        
        Returns
        ----------
        This function does not return anything. It saves the plot values in the results dictionary. 
        """
        class_index = "NA"
        ind=[]
        for i in range(len(self.tran_results['model_list'][model_num]['local_interpretability'])):
            ind.append(self.tran_results['model_list'][model_num]['local_interpretability'][i]['id'])
        if(n not in ind):
            #creating class index and explanation scenarios based on the model type
            if(self.model_params[model_num].model_type)!='regression':                
                if(len(self.model_params[model_num].model_object.classes_)>2):  
                    class_index = list(self.model_params[model_num].model_object.classes_).index(np.array(self.model_params[model_num].y_train)[n-1])   
                    exp = Explanation(self.tran_shap_values[model_num][class_index], 
                        base_values = self.tran_shap_extra[model_num][:,class_index], 
                        data=self.tran_processed_data.values, 
                        feature_names=self.tran_processed_data.columns)
                else:
                    exp = Explanation(self.tran_shap_values[model_num], 
                        base_values = self.tran_shap_extra[model_num], 
                        data=self.tran_processed_data.values, 
                        feature_names=self.tran_processed_data.columns)
            
            else:
                exp = Explanation(self.tran_shap_values[model_num], 
                    base_values = self.tran_shap_extra[model_num], 
                    data=self.tran_processed_data.values, 
                    feature_names=self.tran_processed_data.columns) 


            #plotting the shap waterfall plot for the given index 
            row_index = list(self.tran_processed_data.index).index(n-1)

            #getting the values behind the waterfall plot
            plot_points = pd.DataFrame({
            'Feature_name': self.tran_processed_data.columns,
            'Value': self.tran_processed_data.iloc[row_index],
            'Shap': exp[row_index].values   
            })
            plot_points=plot_points.sort_values(by='Shap', key=abs, ascending = False)
            efx = exp.base_values[row_index]
            fx = (exp.base_values[row_index]) + sum(exp[row_index].values)

            if(self.tran_features==[]):
                pl.figure(constrained_layout = True)
                shap.plots.waterfall(exp[row_index], max_display = self.tran_max_display, show=False)    
                fig=pl.gcf()
                pl.ion()
                pl.close()
                im = self._plot_encode(fig, plot=None)
            
                other_features = pd.DataFrame({'Feature_name':str(len(plot_points[self.tran_max_display-1:])) + " OTHER",'Value' : "" ,'Shap': plot_points[self.tran_max_display-1:][['Shap']].sum().values})
                if(plot_points.shape[0]==self.tran_max_display):
                    local_plot_points = plot_points[:self.tran_max_display]
                    feature_list = local_plot_points[['Feature_name','Value', 'Shap']]
                else:
                    local_plot_points = plot_points[:self.tran_max_display-1]
                    feature_list = pd.concat([local_plot_points,other_features])

                dict_item = {'id':n,'efx': efx,'fx':fx,'feature_info':feature_list[['Feature_name','Value', 'Shap']].to_dict(orient='records')}
                #generating dictionary of values for the indices passed
                self.tran_results['model_list'][model_num]['local_interpretability'].append(dict_item)
                self.tran_results['model_list'][model_num]['plot']['local_plot'][n] = im
                self.tran_results['model_list'][model_num]['plot']['class_index'][n] = class_index
            else:
                local = plot_points[np.isin(plot_points, self.tran_features).any(axis=1)]
                dict_item = {'id':n,'efx': efx,'fx':fx,'feature_info':local[['Feature_name','Value', 'Shap']].to_dict(orient='records')}
                #generating dictionary of values for the indices passed
                self.tran_results['model_list'][model_num]['local_interpretability'].append(dict_item)
                self.tran_results['model_list'][model_num]['plot']['local_plot'][n] = ''
                self.tran_results['model_list'][model_num]['plot']['class_index'][n] = class_index

    def _compute_partial_dependence(self, model_num=0):
        """
        Creates partial dependence plots using sklearn and computes the values for 2 features as required.
        
        Returns
        ----------
        This function does not return anything. It saves the plot values in the results dictionary.
        """   
        top_two = self.tran_top_features[model_num]['Feature_name'].tolist()[:2]
        tran_pdp_feature = self.tran_pdp_feature + top_two

        final_pdp = []
        [final_pdp.append(x) for x in tran_pdp_feature if x not in final_pdp]
        
        self.tran_pdp_feature_list[model_num]=final_pdp[:2]
        
        if(self.model_params[model_num].model_type!='regression'):
            if(self.tran_pdp_target == None) and len(self.model_params[model_num].model_object.classes_)>2:
                self.tran_pdp_target = self.model_params[model_num].pos_label[0]
                
        if(getattr(self.model_params[model_num].model_object, "_estimator_type",None) not in ["classifier","regressor"]):
            if(self.model_params[model_num].model_type == 'regression'):
                setattr(self.model_params[model_num].model_object, "_estimator_type", "regressor")
            else:
                setattr(self.model_params[model_num].model_object, "_estimator_type", "classifier")              
        
        for i in self.tran_pdp_feature_list[model_num]:
            if(type(self.tran_shap_values[model_num])==list):
                PartialDependenceDisplay.from_estimator(self.model_params[model_num].model_object, self.tran_processed_data, [i], target = self.tran_pdp_target)   
            else:
                PartialDependenceDisplay.from_estimator(self.model_params[model_num].model_object, self.tran_processed_data, [i])

            f=pl.gcf() 
            pl.tight_layout()
            im,image_base_64 = self._plot_encode(f, plot="pdp_plot")
            pl.close()
            self.tran_results['model_list'][model_num]['plot']['pdp_plot'][i] = im 
            self.tran_results['model_list'][model_num]['partial_dependence_plot'][i] = image_base_64

        print('{:5s}{:35s}{:<10}'.format('','Partial dependence','done'))

    def _compute_permutation_importance(self,model_num=0):
        """
        Computes permutation importance of each of the features from the process data. 
        Normalize the importance score to get relative percentages for the chart.
 
        Returns
        ----------
        This function does not return anything. It saves the plot values in the results dictionary.
        """
        eval_pbar = tqdm(total=100, desc='Computing Permutation Importance', bar_format='{l_bar}{bar}')
        eval_pbar.update(5)
        if self.perf_metric_obj is None:
            self.perf_metric_obj = PerformanceMetrics(self)

        if (self.model_params[0].model_type in ['regression','uplift']):
            score_func = self.perf_metric_obj.map_perf_metric_to_method[self.perf_metric_name]
        else:
            score_func = self.perf_metric_obj.map_perf_metric_to_method_optimized[self.perf_metric_name]           
        
        if self.evaluate_status == 1:
            base_score = self.perf_metric_obj.result['perf_metric_values'][self.perf_metric_name][0]
        else:
            if (self.model_params[0].model_type == 'regression'):
                base_score = score_func(subgrp_y_true = self.model_params[0].y_true,
                                        subgrp_y_pred = self.model_params[0].y_pred)
            elif (self.model_params[0].model_type == 'uplift'):
                y_prob = [model.y_prob for model in self.model_params]
                base_score = score_func(subgrp_y_true = self.model_params[1].y_true,
                                        subgrp_y_prob = y_prob,
                                        subgrp_e_lift = self.e_lift)               
            else:         
                base_score = score_func(subgrp_y_true = self.model_params[0].y_true,
                                        subgrp_y_pred = self.model_params[0].y_pred,
                                        subgrp_y_prob = self.model_params[0].y_prob)
        eval_pbar.update(5)

        permutation_additional = Constants().permutation_additional
        permutation_additional = min(int(self.tran_max_display*(1+permutation_additional)), self.model_params[0].x_train.shape[1])
        
        if(self.tran_features==[]):
            feature_list = self.tran_top_features[model_num]['Feature_name'].tolist()[:permutation_additional]
        else:
            feature_list = self.tran_features
        
        diff = []
        neg_flag = []
        new_list = list(set(feature_list)-set(self.permutation_importance['feature']))
        
        for feature in new_list:
            original = self.model_params[0].x_test[feature].copy()
            transformed = np.array(original).copy()
            np.random.seed(0)
            np.random.shuffle(transformed)
            self.model_params[0].x_test[feature] = transformed
            if (self.model_params[0].model_type == 'regression'):
                y_pred = self.model_params[0].model_object.predict(self.model_params[0].x_test)
                base_score_new = score_func(subgrp_y_true = self.model_params[0].y_true,
                                            subgrp_y_pred = y_pred)                
            elif (self.model_params[0].model_type == 'uplift'):
                y_prob1 = self.model_params[0].model_object.predict_proba(self.model_params[0].x_test)
                y_prob2 = self.model_params[1].model_object.predict_proba(self.model_params[0].x_test)
                for i in range(y_prob1.shape[0]):
                    y_prob1[i]=y_prob1[i][::-1]
                    y_prob2[i]=y_prob2[i][::-1]
                y_prob = [y_prob1,y_prob2]
                e_lift = self._get_e_lift(y_pred_new=y_prob[1])
                base_score_new = score_func(subgrp_y_true = self.model_params[1].y_true,
                                            subgrp_y_prob = y_prob,
                                            subgrp_e_lift = e_lift)
            else:
                y_pred = self.model_params[0].model_object.predict(self.model_params[0].x_test)
                y_prob = self.model_params[0].model_object.predict_proba(self.model_params[0].x_test)
                if y_prob.shape[1] > 2 or self.model_params[0].model_object.classes_[1] != 1:
                    y_pred, pos_label = self._check_label(y_pred, self.model_params[0].pos_label, self.model_params[0].neg_label, obj_in=self.model_params[0], y_pred_flag=True)
                    y_prob = process_y_prob(self.model_params[0].model_object.classes_ , y_prob, 
                                            self.model_params[0].pos_label, self.model_params[0].neg_label)
                else:
                    y_prob = self.model_params[0].model_object.predict_proba(self.model_params[0].x_test)[:,1]
                base_score_new = score_func(subgrp_y_true = self.model_params[0].y_true,
                                            subgrp_y_pred = y_pred,
                                            subgrp_y_prob = y_prob)
            self.model_params[0].x_test[feature] = original
            diff.append(abs(base_score-base_score_new))
            if self.perf_metric_name in ['log_loss','rmse','mape','wape']:
                if base_score_new > base_score:
                    neg_flag.append(0)
                else:
                    neg_flag.append(1)
            else:
                if base_score > base_score_new:
                    neg_flag.append(0)
                else:
                    neg_flag.append(1)
            eval_pbar.update(80/len(new_list))
        
        self.permutation_importance = pd.concat([self.permutation_importance, pd.DataFrame({'feature': new_list,'diff': diff,'neg_flag': neg_flag})])
        self.permutation_importance = self.permutation_importance.sort_values(by='diff', ascending = False) 
        perm_imp = self.permutation_importance.copy()
        perm_imp['score'] = perm_imp['diff']/perm_imp['diff'].max()*100
        perm_imp = perm_imp[:self.tran_max_display]
        neg_feature_list = list(perm_imp.loc[perm_imp['neg_flag'] == 1, 'feature'])
        perm_imp['feature'] = perm_imp.apply(lambda row: row['feature'] + '*' if row['neg_flag'] == 1 else row['feature'], axis=1)
        self.tran_results['permutation_score'] = perm_imp[['feature','score']].to_dict(orient='records')
        self.tran_results['permutation_score'].insert(0,{'title': "Permutation Importance Plot based on |Metric_old - Metric_new|"})
        
        pl.figure(constrained_layout = True)
        pl.barh(y=perm_imp.feature, width=perm_imp.score,height=0.5)
        pl.gca().invert_yaxis()
        if len(neg_feature_list) > 0:
            if self.perf_metric_name in ['log_loss','rmse','mape','wape']:
                pl.xlabel('* indicates $Metric_{old} > Metric_{new}$', fontsize = 8, loc='left')
                self.tran_results['permutation_score'].append({'footnote': '* indicates Metric_old > Metric_new'})
            else:
                pl.xlabel('* indicates $Metric_{old} < Metric_{new}$', fontsize = 8, loc='left')
                self.tran_results['permutation_score'].append({'footnote': '* indicates Metric_old < Metric_new'})
        else:
            self.tran_results['permutation_score'].append({'footnote': None})

        fig=pl.gcf()
        pl.close()
        im = self._plot_encode(fig, plot=None)
        self.tran_results['model_list'][0]['plot']['perm_plot'] = im
        eval_pbar.set_description('Permutation Importance Calculated')
        eval_pbar.update(100 - eval_pbar.n)
        eval_pbar.close()

        print('{:5s}{:35s}{:<10}'.format('','Permutation importance','done'))

    def _plot(self,model_num=0,interpretability_plot_flag=True,pdp_plot_flag=True,perm_imp_plot_flag=True):
        if(interpretability_plot_flag == True):
            summary_plot = self.tran_results['model_list'][model_num]['plot']['summary']
            latest_local_key = list(self.tran_results['model_list'][model_num]['plot']['local_plot'].keys())[-1]
            local_plot = self.tran_results['model_list'][model_num]['plot']['local_plot'][latest_local_key]
        
            if(type(summary_plot)!=str):
            #creating grid for subplots
                fig = pl.figure()
                #fig.set_figheight(10)
                fig.set_figwidth(15)
                ax1 = pl.subplot2grid(shape=(1, 2), loc=(0, 0), colspan=1)
                ax2 = pl.subplot2grid(shape=(1, 2), loc=(0, 1), colspan=1)
                ax1.imshow(summary_plot,aspect='auto')
                ax1.set_title("Global Interpretability Plot",fontsize=12,fontweight='bold')
                ax2.imshow(local_plot,aspect='auto')
                if(self.tran_results['model_list'][model_num]['plot']['class_index'][latest_local_key] != "NA"): 
                    label = self.model_params[model_num].model_object.classes_[self.tran_results['model_list'][model_num]['plot']['class_index'][latest_local_key]]
                    ax2.set_title("Local Interpretability Plot for index = " + str(latest_local_key) + ", target class: " + str(label), fontsize=12,fontweight='bold')
                else:
                    ax2.set_title("Local Interpretability Plot for index = " + str(latest_local_key), fontsize=12,fontweight='bold')
                for i in [ax1,ax2]:
                    i.spines[['top','right','bottom','left']].set_visible(False)
                    i.set_xticks([])
                    i.set_yticks([]) 
                pl.tight_layout(pad=1.0)
                pl.show()
            else:
                summary_plot_datapoints = self.tran_results['model_list'][model_num]['summary_plot']
                summary_plot_datapoints = pd.DataFrame(summary_plot_datapoints)
                local_plot_datapoints = self.tran_results['model_list'][model_num]['local_interpretability'][-1]['feature_info']
                local_plot_datapoints=pd.DataFrame(local_plot_datapoints)
                efx = round(self.tran_results['model_list'][model_num]['local_interpretability'][-1]['efx'],3)
                fx = round(self.tran_results['model_list'][model_num]['local_interpretability'][-1]['fx'],3)
                ids = self.tran_results['model_list'][model_num]['local_interpretability'][-1]['id']
                if(self.tran_results['model_list'][model_num]['plot']['class_index'][latest_local_key] != "NA"): 
                    label = self.model_params[model_num].model_object.classes_[self.tran_results['model_list'][model_num]['plot']['class_index'][latest_local_key]]
                    local_title = 'Local interpretability for index: ' + str(ids) + ', target class: '+ str(label) + '\\nefx: ' + str(efx) + ', fx: ' + str(fx)
                else:
                    local_title = 'Local interpretability for index: ' + str(ids) + '\\nefx: ' + str(efx) + ', fx: ' + str(fx)
                styles = [dict(selector="caption",
                       props=[("text-align", "center"),
                              ("font-size", "12"),
                              ("font-weight",'950'),
                              ("color", 'black')])]
                summary_plot_datapoints = summary_plot_datapoints.style.set_table_attributes("style='display:inline'").set_caption('Global interpretability values').set_table_styles(styles)
                local_plot_datapoints = local_plot_datapoints.style.set_table_attributes("style='display:inline'").set_caption(local_title).set_table_styles(styles)
                display_html(20*"\xa0"+summary_plot_datapoints._repr_html_()+75*"\xa0"+local_plot_datapoints._repr_html_().replace("\\n","<br>"), raw=True)
        
        if(pdp_plot_flag==True):
            pdp_plot1 = (self.tran_results['model_list'][model_num]['plot']['pdp_plot'][self.tran_pdp_feature_list[model_num][0]])
            pdp_plot2 = (self.tran_results['model_list'][model_num]['plot']['pdp_plot'][self.tran_pdp_feature_list[model_num][1]])
            fig = pl.figure()
            fig.set_figheight(14)
            fig.set_figwidth(12)
            ax3=pl.subplot2grid(shape=(1, 2), loc=(0, 0), colspan=1)
            ax4=pl.subplot2grid(shape=(1, 2), loc=(0, 1), colspan=1)
            ax3.imshow(pdp_plot1)
            ax4.imshow(pdp_plot2)   
            if(self.model_params[0].model_type != 'regression') and (len(self.model_params[model_num].model_object.classes_)>2):
                ax3.set_title("Partial Dependence Plot for " + str(self.tran_pdp_feature_list[model_num][0]) + "(class " + str(self.tran_pdp_target) + ")" ,fontsize=9,fontweight='bold')
                ax4.set_title("Partial Dependence Plot for " + str(self.tran_pdp_feature_list[model_num][1]) + "(class " + str(self.tran_pdp_target) + ")",fontsize=9,fontweight='bold')
            else: 
                ax3.set_title("Partial Dependence Plot for " + str(self.tran_pdp_feature_list[model_num][0]),fontsize=9,fontweight='bold')
                ax4.set_title("Partial Dependence Plot for " + str(self.tran_pdp_feature_list[model_num][1]),fontsize=9,fontweight='bold')
            for i in [ax3,ax4]:
                i.spines[['top','right','bottom','left']].set_visible(False)
                i.set_xticks([])
                i.set_yticks([])
            pl.tight_layout(pad=0.7)
            pl.show()
        
        if(perm_imp_plot_flag==True):
            perm_plot = self.tran_results['model_list'][0]['plot']['perm_plot']
            fig = pl.figure()
            fig.set_figheight(8)
            fig.set_figwidth(6)
            ax5 = pl.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=2)
            ax5.imshow(perm_plot)
            ax5.set_title("Permutation Importance Plot based on $\mathbf{|Metric_{old} - Metric_{new}|}$",fontsize=9,fontweight='bold')
            ax5.spines[['top','right','bottom','left']].set_visible(False)
            ax5.set_xticks([])
            ax5.set_yticks([]) 
            pl.show()  
              
    def _data_prep(self,model_num=0):
        if(self.tran_flag['data_sampling_flag']==False):
            self._data_sampling()
            self.tran_flag['data_sampling_flag']=True
        self._shap(model_num=model_num)
        self._top_features(model_num=model_num)
        self.tran_features = list(set(self.tran_features))
        if(self.tran_features!=[] and len(self.tran_features)<self.tran_max_display):
            for j in self.tran_top_features[model_num]['Feature_name'].tolist():
                if(j not in self.tran_features):
                    self.tran_features.append(j)
                if(len(self.tran_features)==self.tran_max_display):
                    break

        self.tran_flag[model_num]['data_prep_flag']=True

    def explain(self, disable = None, local_index = None, output = True, model_num = None):
        """
        Compiles output for the entire transparency analysis basis the inputs given by the user.

        Parameters
        ----------
        local_index : int/list, default=None
                It stores the value of the index required to calculate local interpretability.

        output : boolean, default=True
                If output = True, all the transparency plots will be shown to the user on the console.

        force : boolean, default=False
                Stores the binary flag to indicate if the processed dataset needs to be updated.
                It true, the processed data is updated to include the given local index.
                If false, the existing processed data will be used to shown any relevant outputs.

        Returns
        ----------
        This function does not return anything. It will print the charts and compile the final dictionary as per the user input.
        """
        valid_input = ['interpret','partial_dep','perm_imp']
        valid_disable = []
        if disable is not None:
            if(type(disable)!=list):
                self.err.push('type_error', var_name="disable", given=type(disable), expected='list', function_name="explain")
                self.err.pop()
            else:
                for i in disable:
                    if(i in valid_input):
                        valid_disable.append(i)

        disable_flags = {}
        for i in valid_input:
            disable_flags[i] = False
        if disable is not None:
            for i in valid_disable:
                disable_flags[i] = True

        if(model_num is None):
            model_num = len(self.model_params)-1
        elif(type(model_num) != int):
            self.err.push('type_error', var_name="model_num", given=type(model_num), expected='int', function_name="explain")
            self.err.pop()
        elif(model_num not in range(1,len(self.model_params)+1)):
            self.err.push('value_error', var_name = "model_num", given = model_num, expected = "one of the following integers: " + str(list(range(len(self.model_params)+1))[1:]), function_name = "explain")
            self.err.pop()
        else:
            model_num = model_num-1
        
        if type(local_index) in [int,float]:
            local_index = int(local_index)
            if local_index < 1 or local_index > self.model_params[0].x_train.shape[0]:
                self.err.push('value_error', var_name = "local_index", given = local_index, expected = "An integer value within the index range 1-" + str(self.model_params[0].x_train.shape[0]), function_name = "explain")
                self.err.pop()
        elif(local_index is not None):
            self.err.push('type_error', var_name = "local_index", given = type(local_index), expected = "An integer value within the index range 1-" + str(self.model_params[0].x_train.shape[0]), function_name = "explain")
            self.err.pop()

        if(len(valid_disable)>0) and (local_index is not None):
            print("Warning: The local interpretability plot is shown basis the given index and input for disable is ignored.")

        if local_index is None:
            #all disable flags are true
            if(len(list(set(valid_input)-set(valid_disable)))==0):
                print("Skipped: All transparency analysis are disabled for model " + str(model_num+1) + ".")
            else:
                print('{:40s}{:<10}'.format('Running transparency for model ' + str(model_num+1), 'done'))
                if(self.tran_flag[model_num]['data_prep_flag']==False):
                    self._data_prep(model_num=model_num)
                    print('{:5s}{:35s}{:<10}'.format('','Data preparation','done'))
                if(disable_flags['interpret']==False):
                    if(self.tran_flag[model_num]['interpret']==False):
                        self._global(model_num=model_num)
                        for idx in self.tran_index:
                            self._local(n=idx, model_num=model_num)
                            self.tran_flag[model_num]['interpret']=True
                        print('{:5s}{:35s}{:<10}'.format('','Interpretability','done'))

                if(disable_flags['partial_dep']==False):
                    if(self.tran_flag[model_num]['partial_dep']==False):
                        self._compute_partial_dependence(model_num=model_num)
                        self.tran_flag[model_num]['partial_dep']=True

                if(disable_flags['perm_imp']==False):
                    if(self.tran_flag[model_num]['perm_imp']==False):
                        self._compute_permutation_importance(model_num=model_num)
                        self.tran_flag[model_num]['perm_imp']=True

                if output == True:
                    self._plot(model_num = model_num,interpretability_plot_flag=not disable_flags['interpret'],pdp_plot_flag=not disable_flags['partial_dep'],perm_imp_plot_flag=not disable_flags['perm_imp'])
            
        else:
            if self.tran_flag[model_num]['interpret'] == False:
                if(self.tran_flag[model_num]['data_prep_flag'] == False):
                    self.tran_index.append(local_index)
                    self._data_prep(model_num=model_num)
                    print('{:5s}{:35s}{:<10}'.format('','Data preparation','done'))
                elif local_index-1 in self.tran_processed_data.index:
                    self.tran_index.append(local_index)
                else:
                    print("Warning: Given value of local_index not found in the processed dataset. Please re-initiliaze the use case object with tran_index as required or try with following indices:")
                    print((np.sort(np.array(self.tran_processed_data.index)))[:10]+1)
                self._global(model_num=model_num)
                for idx in self.tran_index:
                    self._local(n=idx, model_num=model_num)
                print('{:5s}{:35s}{:<10}'.format('','Interpretability','done'))
                local_plot = self.tran_results['model_list'][model_num]['plot']['local_plot'][local_index]
                if(type(local_plot)!=str):
                    fig = pl.figure()
                    fig.set_figheight(8)
                    fig.set_figwidth(6)
                    ax1 = pl.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1)
                    ax1.imshow(local_plot)
                    if(self.tran_results['model_list'][model_num]['plot']['class_index'][local_index] != "NA"): 
                        label = self.model_params[model_num].model_object.classes_[self.tran_results['model_list'][model_num]['plot']['class_index'][local_index]]
                        ax1.set_title("Local plot for Model " + str(model_num + 1) + " for index = " + str(local_index) + ", target class: " + str(label),fontsize=9,fontweight='bold')
                    else:
                        ax1.set_title("Local plot for Model " + str(model_num + 1) + " for index = " + str(local_index),fontsize=9,fontweight='bold')
                    ax1.spines[['top','right','bottom','left']].set_visible(False)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    pl.show()
                else:
                    local_plot_datapoints=self.tran_results['model_list'][model_num]['local_interpretability'][-1]['feature_info']
                    local_plot_datapoints=pd.DataFrame(local_plot_datapoints)
                    efx = round(self.tran_results['model_list'][model_num]['local_interpretability'][-1]['efx'],3)
                    fx = round(self.tran_results['model_list'][model_num]['local_interpretability'][-1]['fx'],3)
                    ids = self.tran_results['model_list'][model_num]['local_interpretability'][-1]['id']
                    if(self.tran_results['model_list'][model_num]['plot']['class_index'][local_index] != "NA"): 
                        label = self.model_params[model_num].model_object.classes_[self.tran_results['model_list'][model_num]['plot']['class_index'][local_index]]
                        local = 'Local interpretability for index: ' + str(ids) + ', target class: '+ str(label) + '\\n[efx: ' + str(efx) + ', fx: ' + str(fx) + ']'
                    else:
                        local = 'Local interpretability for index: ' + str(ids) + '\\n[efx: ' + str(efx) + ', fx: ' + str(fx) + ']'
                    styles = [dict(selector="caption",
                        props=[("text-align", "center"),
                                ("font-size", "12"),
                                ("font-weight",'950'),
                                ("color", 'black')])]
                    local_plot_datapoints = local_plot_datapoints.style.set_table_attributes("style='display:inline'").set_caption(local).set_table_styles(styles)
                    display_html(40*"\xa0"+local_plot_datapoints._repr_html_().replace("\\n","<br>"), raw=True)
                self.tran_flag[model_num]['interpret']=True  
            
            else:
                if local_index-1 in self.tran_processed_data.index:
                    self._local(n=local_index, model_num=model_num)
                    local_plot = self.tran_results['model_list'][model_num]['plot']['local_plot'][local_index]
                    fig = pl.figure()
                    fig.set_figheight(8)
                    fig.set_figwidth(6)
                    if(type(local_plot)!=str):
                        ax1 = pl.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1)
                        ax1.imshow(local_plot)
                        if(self.tran_results['model_list'][model_num]['plot']['class_index'][local_index] != "NA"): 
                            label = self.model_params[model_num].model_object.classes_[self.tran_results['model_list'][model_num]['plot']['class_index'][local_index]]
                            ax1.set_title("Local plot for Model " + str(model_num + 1) + " for index = " + str(local_index) + ", target class: " + str(label),fontsize=9, fontweight='bold')
                        else:
                            ax1.set_title("Local plot for Model " + str(model_num + 1) + " for index = " + str(local_index),fontsize=9, fontweight='bold')
                        ax1.spines[['top','right','bottom','left']].set_visible(False)
                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        pl.show()
                    else:
                        local_plot_datapoints=self.tran_results['model_list'][model_num]['local_interpretability'][-1]['feature_info']
                        local_plot_datapoints=pd.DataFrame(local_plot_datapoints)
                        efx = round(self.tran_results['model_list'][model_num]['local_interpretability'][-1]['efx'],3)
                        fx = round(self.tran_results['model_list'][model_num]['local_interpretability'][-1]['fx'],3)
                        ids = self.tran_results['model_list'][model_num]['local_interpretability'][-1]['id']
                        if(self.tran_results['model_list'][model_num]['plot']['class_index'][local_index] != "NA"): 
                            label = self.model_params[model_num].model_object.classes_[self.tran_results['model_list'][model_num]['plot']['class_index'][local_index]]
                            local = 'Local interpretability for index: ' + str(ids) + ', target class: '+ str(label) + '\\n[efx: ' + str(efx) + ', fx: ' + str(fx) + ']'
                        else:
                            local = 'Local interpretability for index: ' + str(ids) + '\\n[efx: ' + str(efx) + ', fx: ' + str(fx) + ']'
                        styles = [dict(selector="caption",
                            props=[("text-align", "center"),
                                    ("font-size", "12"),
                                    ("font-weight",'950'),
                                    ("color", 'black')])]
                        local_plot_datapoints = local_plot_datapoints.style.set_table_attributes("style='display:inline'").set_caption(local).set_table_styles(styles)
                        display_html(40*"\xa0"+local_plot_datapoints._repr_html_().replace("\\n","<br>"), raw=True)
                else:
                    print("Warning: Given value of local_index not found in the processed dataset. Please re-initiliaze the use case object with tran_index as required or try with following indices:")
                    print((np.sort(np.array(self.tran_processed_data.index)))[:10]+1)

    def _tran_compile(self,disable=[]):

        """
        Ensures tran results dictionary is udpated with all the results.
        
        Returns
        ----------
        This function returns the transparency results to be included in the json file.
        """

        for i in range(len(self.model_params)):
            self.explain(disable=disable,model_num=i+1,output= False)
        tran_results = copy.deepcopy(self.tran_results)
        if (len(list(set({'interpret','partial_dep','perm_imp'})-set(disable)))==0):
            tran_results = None
        else:
            if(tran_results['permutation_score']==''):
                tran_results['permutation_score'] = None
            for i in range(len(self.model_params)):
                del tran_results['model_list'][i]['plot']
                if(len(tran_results['model_list'][i]['local_interpretability']))==0:
                    tran_results['model_list'][i]['local_interpretability'] = None
                if(tran_results['model_list'][i]['partial_dependence_plot'])=={}:
                    tran_results['model_list'][i]['partial_dependence_plot'] = None
                if(tran_results['model_list'][i]['summary_plot'])=='':
                    tran_results['model_list'][i]['summary_plot'] = None    
        return tran_results    

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

        y_bin = y
        if y_pred_flag == True and obj_in.unassigned_y_label[0]:
            y_bin = check_data_unassigned(obj_in, y_bin, y_pred_negation_flag=True)
            
        else:
            row = np.isin(y_bin, pos_label)
            if sum(row) == len(y_bin) :
                err_.append(['value_error', "pos_label", pos_label, "not all y_true labels"])
            elif sum(row) == 0 :
                err_.append(['value_error', "pos_label", pos_label, set(y_bin)])            
            for i in range(len(err_)):
                err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                        function_name="_check_label")
            y_bin[row] = 1 
            y_bin[~row] = 0
        
        pos_label2 = [[1]]
        y_bin = y_bin.astype(np.int8)
            
        if y_bin.dtype.kind in ['i']:
            y_bin  = y_bin.astype(np.int8)

        err.pop()

        return y_bin, pos_label2