#!/usr/bin/python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
#
# Copyright (c) 2019 kuaishou.com, Inc. All Rights Reserved
#
# -------------------------------------------------------------------------------

"""
@author: benyuansong
"""

import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


class preprocessing:
    
    
    
    def did_preprocessing(data, treatment_col = 'treatment', name_col = 'user_id', date_col = 'date'):
        """
        drop the duplicates data by username and date. change NA and -124 to 0
        """
        
        if type(data) != pd.core.frame.DataFrame:
            raise TypeError("Type of panel data should be dataframe")
        if type(treatment_col) != str:
            raise TypeError("Type of treatment_col should be string")
        if type(name_col) != str:
            raise TypeError("Type of name_col should be string")
        if type(date_col) != str:
            raise TypeError("Type of date_col should be string")
            
        data.drop_duplicates([name_col,date_col], inplace = True)
        data.reset_index(drop=True, inplace = True)
        data.fillna(0, inplace=True)
        data.rename(columns={name_col:'user_id', treatment_col:'treatment', date_col:'date'}, inplace = True)
        
        for name in data.columns:
            if data[name].dtypes == int:
                if sorted(list(data[name].value_counts().index)) == [-124,0,1]:
                    data[name] = data[name].apply(lambda x : 0 if x==-124 else x)
        
        return data
    
    
    def ttest_preprocessing(data, y, treatment_col = 'treatment', name_col = 'user_id',basename=['base1','base2'],expname='exp1'):
        """
        calculate the average of giving y group by userid then output base data and exp data
        """
        
        if type(data) != pd.core.frame.DataFrame:
            raise TypeError("Type of panel data should be dataframe")
        if type(treatment_col) != str:
            raise TypeError("Type of treatment_col should be string")
        if type(name_col) != str:
            raise TypeError("Type of name_col should be string")
        if type(y) != str and type(y) != list:
            raise TypeError("Type of y should be string or list")
        
        data.fillna(0, inplace=True)
        data.rename(columns={name_col:'user_id', treatment_col:'treatment'}, inplace = True)
        
        if type(y) == list:
            dict = {}
            for m in y:
                dict[m]='mean'
            temp = data.groupby('user_id').agg(dict)
            temp['user_id']=temp.index
            temp.reset_index(drop=True, inplace = True)
            data.drop_duplicates(['user_id'], inplace = True)
            data.drop(y,axis=1,inplace = True)
            data = data.merge(temp,on = 'user_id')
        
        else:
            temp = data.groupby('user_id').agg({y:'mean'})
            temp['user_id']=temp.index
            temp.reset_index(drop=True, inplace = True)
            data.drop_duplicates(['user_id'], inplace = True)
            data.drop([y],axis=1,inplace = True)
            data = data.merge(temp,on = 'user_id')
        
        return data[data[treatment_col]==basename[0]],data[data[treatment_col]==basename[1]],data[data[treatment_col]==expname]
        




class did:
    
    
    
    def algorithm_show(data, last_feature_col, by_treatment=True, label='treatment'):
        """
        check the score of used algorithms
        """
        
        temp=data.copy()
        temp.drop_duplicates(['user_id','treatment'], inplace = True)
        temp.reset_index(drop=True, inplace = True)
        temp_treatment=temp[temp.treatment==1]
        
        if by_treatment:
            temp_X = temp.iloc[:,:last_feature_col].drop(['user_id','date','treatment'],axis=1)
            temp_X = pd.get_dummies(temp_X)
            X_train, X_test, y_train, y_test = train_test_split(temp_X, temp['treatment'], test_size=0.25, random_state=0, stratify=temp['treatment'])
        else:
            temp_X = temp.iloc[:,:last_feature_col].drop(['user_id','date',label],axis=1)
            temp_X = pd.get_dummies(temp_X)
            X_train, X_test, y_train, y_test = train_test_split(temp_X[temp_X.treatment==1].iloc[:,1:], temp_treatment.loc[:,[label]], test_size=0.25, random_state=0, stratify=temp_treatment.loc[:,[label]])
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=6,class_weight='balanced',random_state=0)
        rf_model.fit(X_train,y_train)
        lr_model = LogisticRegression(class_weight='balanced',random_state=0)
        lr_model.fit(X_train,y_train)
        lgb_model = LGBMClassifier(max_depth=6,class_weight='balanced',random_state=0)
        lgb_model.fit(X_train,y_train)
        n=len(y_train[y_train[label]==0])/len(y_train[y_train[label]==1])
        xgb_model = XGBClassifier(max_depth=6,scale_pos_weight=n,random_state=0,importance_type='weight')
        xgb_model.fit(X_train,y_train)
        
        def pred_result(model,x,y):
            ypred = model.predict(x)
            pscore = model.predict_proba(x)[:,1]
            accuracy = metrics.accuracy_score(y, ypred)
            fpr, tpr, thresholds = metrics.roc_curve(y, pscore)
            auc_score = metrics.auc(fpr, tpr)
            precision = metrics.precision_score(y, ypred)
            precision_0 = metrics.precision_score(y, ypred,pos_label=0)
            recall = metrics.recall_score(y, ypred)
            recall_0 = metrics.recall_score(y, ypred,pos_label=0)
            f1score = metrics.f1_score(y, ypred)
            f1score_0 = metrics.f1_score(y, ypred,pos_label=0)
            return [accuracy,auc_score,precision,recall,f1score,precision_0,recall_0,f1score_0]
        
        result = pd.DataFrame([],index=['accuracy','auc_score','precision','recall','f1score','precision_0','recall_0','f1score_0'])
        
        result['RandomForest'] = pred_result(rf_model,X_test,y_test)
        result['LogisticRegression'] = pred_result(lr_model,X_test,y_test)
        result['LightGBM'] = pred_result(lgb_model,X_test,y_test)
        result['XGBoost'] = pred_result(xgb_model,X_test,y_test)
        
        return result
    
    
    def matching(data, last_feature_col, model='LightGBM', by_treatment=True ,label='treatment', caliper=0.05,top=10):
        """
        Automatically match control group with treatment groups by propensity score matching
        """
        
        if model not in ['RandomForest','LogisticRegression','LightGBM','XGBoost']:
            raise NameError("We only support 'RandomForest','LogisticRegression','LightGBM' and 'XGBoost' models for now!")
        
        temp=data.copy()
        temp.drop_duplicates(['user_id','treatment'], inplace = True)
        temp.reset_index(drop=True, inplace = True)
        temp_treatment=temp[temp.treatment==1]
        
        if by_treatment:
            temp_X = temp.iloc[:,:last_feature_col].drop(['user_id','date','treatment'],axis=1)
            temp_X = pd.get_dummies(temp_X)
            X_train = temp_X
            y_train = temp['treatment']
        else:
            temp_X = temp.iloc[:,:last_feature_col].drop(['user_id','date',label],axis=1)
            temp_X = pd.get_dummies(temp_X)
            X_train, y_train = temp_X[temp_X.treatment==1].iloc[:,1:], temp_treatment.loc[:,[label]]
        
        feature_importance = pd.DataFrame([])
        feature_importance['column']=X_train.columns
        
        if model == 'RandomForest':
            propensity_model = RandomForestClassifier(n_estimators=100,max_depth=6,class_weight='balanced',random_state=0)
            propensity_model.fit(X_train,y_train)
            feature_importance['feature_importance']=propensity_model.feature_importances_
            feature_importance=feature_importance.sort_values(by='feature_importance',ascending=False).iloc[:top,:]
            
        elif model == 'LogisticRegression':
            propensity_model = LogisticRegression(class_weight='balanced',random_state=0)
            propensity_model.fit(X_train,y_train)
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            scale_data = scaler.transform(X_train)
            scale_data = pd.DataFrame(scale_data,columns=X_train.columns)
            scale_model = LogisticRegression(class_weight='balanced',random_state=0)
            scale_model.fit(scale_data,y_train)
            feature_importance['feature_importance']=scale_model.coef_[0]
            feature_importance['feature_abs']=feature_importance.feature_importance.apply(abs)
            feature_importance=feature_importance.sort_values(by='feature_abs',ascending=False).iloc[:top,:]
            
        elif model == 'LightGBM':
            propensity_model = LGBMClassifier(max_depth=6,class_weight='balanced',random_state=0)
            propensity_model.fit(X_train,y_train)
            feature_importance['feature_importance']=propensity_model.feature_importances_
            feature_importance=feature_importance.sort_values(by='feature_importance',ascending=False).iloc[:top,:]
        else:
            n=len(y_train[y_train[label]==0])/len(y_train[y_train[label]==1])
            propensity_model = XGBClassifier(max_depth=6,scale_pos_weight=n,random_state=0,importance_type='weight')
            propensity_model.fit(X_train,y_train)
            feature_importance['feature_importance']=propensity_model.feature_importances_
            feature_importance=feature_importance.sort_values(by='feature_importance',ascending=False).iloc[:top,:]
        
        if by_treatment:
            pscore = propensity_model.predict_proba(X_train)[:,1]
            pscore = pd.Series(pscore)

            N0, N1 = temp[temp.treatment == 0].index, temp[temp.treatment == 1].index
            g0, g1 = pscore[temp.treatment == 0], pscore[temp.treatment == 1]
        else:
            pscore = propensity_model.predict_proba(temp_X.iloc[:,1:])[:,1]
            pscore = pd.Series(pscore)

            N0, N1 = temp[temp.treatment == 0].index, temp[(temp.treatment == 1)&(temp[label] == 1)].index
            g0, g1 = pscore[temp.treatment == 0], pscore[(temp.treatment == 1)&(temp[label] == 1)]
 
        
        if len(N0) < len(N1):
            N0, N1, g0, g1 = N1, N0, g1, g0

        order = np.random.permutation(N1)
        matches = {}
        
        for m in order:
                
            dif = abs(g1[m] - g0)
            
            if dif.min() <= 0.05:
                matches[m] = [dif.idxmin()]
        
        row = list(set([m for match in matches.values() for m in match])) + [m for m in matches.keys()]
        match_id = temp.ix[row, 'user_id']

        matched = data[data['user_id'].isin(match_id)]
        
        return matched,pscore,match_id,feature_importance
    
    
    def plotMatchedSample(data,pscore,match_id,by_treatment=True,label='treatment'):
        """
        output Propensity Score Distribution Graph by Matching
        """
        
        temp = data.copy()
        temp.drop_duplicates(['user_id','treatment'], inplace = True)
        temp.reset_index(drop=True, inplace = True)
        plt.figure(figsize=(16,6))
        plt.subplot(121)
        if by_treatment:
            sns.kdeplot(pscore[temp.treatment == 1], color = "green", shade = True, label = "exp")
            sns.kdeplot(pscore[(temp.treatment == 1)&(temp['user_id'].isin(match_id))], color = "yellow", shade = True, label = "exp after matching")
        else:
            sns.kdeplot(pscore[(temp.treatment == 1)&(temp[label]==1)], color = "green", shade = True, label = "exp:"+label+"==1")
            sns.kdeplot(pscore[(temp.treatment == 1)&(temp[label]==0)], color = "yellow", shade = True, label = "exp:"+label+"==0")
        sns.kdeplot(pscore[temp.treatment == 0], color = "blue", shade = True, label = "control")
        sns.kdeplot(pscore[(temp.treatment == 0)&(temp['user_id'].isin(match_id))], color = "red", shade = True, label = "control after matching")
        plt.title('Propensity Score Distribution by Matching')
        plt.xlabel('Propensity Score')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(122)
        if by_treatment:
            temp1 = pd.Series([pscore[temp.treatment == 0].count(),pscore[temp.treatment == 1].count(),pscore[(temp.treatment == 0)&(temp['user_id'].isin(match_id))].count(),pscore[(temp.treatment == 1)&(temp['user_id'].isin(match_id))].count()])
            patches = plt.bar(np.arange(4),height=temp1,width=0.5)
            plt.xticks(np.arange(4),('before_match_control','before_match_exp','after_match_control','after_match_exp'))
        else:
            temp1 = pd.Series([pscore[temp.treatment == 0].count(),pscore[(temp.treatment == 1)&(temp[label]==1)].count(),pscore[(temp.treatment == 1)&(temp[label]==0)].count(),pscore[(temp.treatment == 0)&(temp['user_id'].isin(match_id))].count(),pscore[(temp.treatment == 1)&(temp[label]==1)&(temp['user_id'].isin(match_id))].count()])
            patches = plt.bar(np.arange(5),height=temp1,width=0.5)
            plt.xticks(np.arange(5),('before_match_control','before_match_exp_label_1','before_match_exp_label_0','after_match_control','after_match_exp_label_1'),rotation=60)      
        for rect in patches:
            height = rect.get_height()
            if height != 0:
                plt.text(rect.get_x() + rect.get_width()/2, height ,'{:.0f}'.format(height),ha='center',va='bottom')
        plt.title('Sample Count Information')
        plt.ylabel('User Count')
        plt.legend()
    
    
    def check_parallel(dataframe, y, post, step = 1):
        """
        output parallel check graph
        """
        
        if type(y) != str and type(y) != list:
            raise TypeError("Type of y should be string or list")
        if type(y) == str:
            temp = dataframe.copy()
            add = pd.get_dummies(temp.date)    
            add.columns = map(lambda x : "date_" + str(x), add.columns)
            date_lst = dataframe.date.astype('str').unique().tolist()   
            date_lst.sort()
            date_ = date_lst[date_lst.index(post)-1]
            add.drop("date_" + str(date_), axis = 1, inplace = True)
            outcome = temp[y]
            base=pd.DataFrame([])
            check_name=[]
            for name in add.columns:
                base["inter_" + str(name) + "_treatment"] = add[name]*temp.treatment
                check_name.append("inter_" + str(name) + "_treatment")
    
            base = pd.concat([base, add], axis = 1)
            base = pd.concat([base, temp.treatment], axis = 1)
            base = pd.concat([base, outcome], axis = 1)
    
            X=base.iloc[:,:-1]
            Y=base[y].map(lambda x : np.log(x+1))
    
            model = sm.OLS(Y,sm.add_constant(X)).fit()
    
            xvalue = list(map(lambda x : x[5:], add.columns))
            x = range(len(xvalue))
            param, se = [], []
            param.extend(list(model.params[check_name]))
            se.extend(list(model.bse[check_name]))
            se = np.array(list(map(lambda x : x*1.96, se)))
        
            plt.errorbar(x, param, se,  marker = '^', capsize = 5, elinewidth = 2, markeredgewidth = 2)
            plt.xticks(x[::step], xvalue[::step],rotation=60)
            plt.title('parallel check of '+y)
            plt.grid(True)
        
        else:
            for name in y:
                temp = dataframe.copy()
                add = pd.get_dummies(temp.date)    
                add.columns = map(lambda x : "date_" + str(x), add.columns)
                date_lst = dataframe.date.astype('str').unique().tolist()   
                date_lst.sort()
                date_ = date_lst[date_lst.index(post)-1]
                add.drop("date_" + str(date_), axis = 1, inplace = True)
                outcome = temp[name]
                base=pd.DataFrame([])
                check_name=[]
                for name1 in add.columns:
                    base["inter_" + str(name1) + "_treatment"] = add[name1]*temp.treatment
                    check_name.append("inter_" + str(name1) + "_treatment")
    
                base = pd.concat([base, add], axis = 1)
                base = pd.concat([base, temp.treatment], axis = 1)
                base = pd.concat([base, outcome], axis = 1)
    
                X=base.iloc[:,:-1]
                Y=base[name].map(lambda x : np.log(x+1))
    
                model = sm.OLS(Y,sm.add_constant(X)).fit()
    
                xvalue = list(map(lambda x : x[5:], add.columns))
                x = range(len(xvalue))
                param, se = [], []
                param.extend(list(model.params[check_name]))
                se.extend(list(model.bse[check_name]))
                se = np.array(list(map(lambda x : x*1.96, se)))
                
                
                plt.errorbar(x, param, se,  marker = '^', capsize = 5, elinewidth = 2, markeredgewidth = 2)
                plt.xticks(x[::step], xvalue[::step],rotation=60)
                plt.title('parallel check of '+name)
                plt.grid(True)
                plt.show()
                plt.close()
        
    
    def check_covariate_balance(data,column):
        """
        output balance check of giving column
        """
        
        data = pd.get_dummies(data)
        def imbalance(str,data):
            return data[data.treatment==1][str].mean(),data[data.treatment==0][str].mean(),abs(data[data.treatment==1][str].mean()-data[data.treatment==0][str].mean()),abs(data[data.treatment==1][str].mean()-data[data.treatment==0][str].mean())/np.sqrt((data[data.treatment==1][str].var()+data[data.treatment==0][str].var())/2)
        
        temp = pd.DataFrame([],columns=['treated','control','dif','bias'])
        for name in column:
            temp.loc[str(name)]=list(imbalance(str(name),data))
        
        return temp
    
    def result(data,y,post):
        """
        output did result
        """
        
        if type(y) != str and type(y) != list:
            raise TypeError("Type of y should be string or list")
        
        temp = data.copy()
        temp['post']=1*(temp['date'].astype('str')>=post)
        
        if type(y) == str:
            temp[y] = temp[y].map(lambda x : np.log(x+1))
            model = smf.ols(y + " ~ post*treatment", data = temp).fit(cov_type='cluster', cov_kwds={'groups': temp['user_id']})
            res = ''
            if model.pvalues["post:treatment"]>0.05:
                res = '不显著'
            else:
                res = '显著'
            print(y+': '+str("%.2f%%" % (model.params["post:treatment"]*100))+' '+str(model.pvalues["post:treatment"])+' '+res)
        
        else:
            for name in y:
                temp[name] = temp[name].map(lambda x : np.log(x+1))
                model = smf.ols(name + " ~ post*treatment", data = temp).fit(cov_type='cluster', cov_kwds={'groups': temp['user_id']})
                res = ''
                if model.pvalues["post:treatment"]>0.05:
                    res = '不显著'
                else:
                    res = '显著'
                print(name+': '+str("%.2f%%" % (model.params["post:treatment"]*100))+' '+str(model.pvalues["post:treatment"])+' '+res)
    

    
class ttest:
    
    
    
    def ttest(base1,base2,exp,y,method=None):
        """
        output ttest result
        """
        
        if type(y) != str and type(y) != list:
            raise TypeError("Type of y should be string or list")
            
        if type(y) == str:
            if method == 'log':
                base1[y] = base1[y].map(lambda x : np.log(x+1))
                base2[y] = base2[y].map(lambda x : np.log(x+1))
                exp[y] = exp[y].map(lambda x : np.log(x+1))
        
            if stats.levene(base1[y], base2[y])[1] >= 0.05:
                aa_pvalue = stats.ttest_ind(base1[y],base2[y])[1]
            else:
                aa_pvalue = stats.ttest_ind(base1[y],base2[y],equal_var = False)[1]
        
            if aa_pvalue < 0.05:
                print(y+": aa test didn't pass because of p_value is {}, that base1 base2 is significantly different.".format(aa_pvalue))
            else:
                if stats.levene(base1[y], exp[y])[1] >= 0.05:
                    ab_pvalue = stats.ttest_ind(base1[y],exp[y])[1]
                else:
                    ab_pvalue = stats.ttest_ind(base1[y],exp[y],equal_var = False)[1]
            
                if method == 'log':
                    difference = np.mean(exp[y])-np.mean(base1[y])
                else:
                    difference = (np.mean(exp[y])-np.mean(base1[y]))/np.mean(base1[y])
            
                print(y+": aa_pvalue:{},ab_pvalue:{},difference:{}".format(aa_pvalue,ab_pvalue,difference))
            
        else:
            for name in y:
                if method == 'log':
                    base1[name] = base1[name].map(lambda x : np.log(x+1))
                    base2[name] = base2[name].map(lambda x : np.log(x+1))
                    exp[name] = exp[name].map(lambda x : np.log(x+1))
        
                if stats.levene(base1[name], base2[name])[1] >= 0.05:
                    aa_pvalue = stats.ttest_ind(base1[name],base2[name])[1]
                else:
                    aa_pvalue = stats.ttest_ind(base1[name],base2[name],equal_var = False)[1]
        
                if aa_pvalue < 0.05:
                    print(name+": aa test didn't pass because of p_value is {}, that base1 base2 is significantly different.".format(aa_pvalue))
                else:
                    if stats.levene(base1[name], exp[name])[1] >= 0.05:
                        ab_pvalue = stats.ttest_ind(base1[name],exp[name])[1]
                    else:
                        ab_pvalue = stats.ttest_ind(base1[name],exp[name],equal_var = False)[1]
            
                    if method == 'log':
                        difference = np.mean(exp[name])-np.mean(base1[name])
                    else:
                        difference = (np.mean(exp[name])-np.mean(base1[name]))/np.mean(base1[name])
            
                    print(name+": aa_pvalue:{},ab_pvalue:{},difference:{}".format(aa_pvalue,ab_pvalue,difference))
            