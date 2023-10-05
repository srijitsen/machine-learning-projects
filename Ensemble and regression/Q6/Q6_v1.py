import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
from numpy import mean
from numpy import std

# Enter You Name Here
myname = "srijit sen"
lib_path = "D:/MDS related Documents/FOML/Assignment 3/Q6/"
os.chdir(lib_path)

def load_data():
    training_data=pd.read_csv('loan_train.csv')
    
    # Load test data set
    test_data=pd.read_csv('loan_test.csv')
     
    print('data loaded')
    return training_data,test_data

def null_values(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
 
def missing_values(X):
    Missing_values_columns=null_values(X)
    Missing_values_columns=Missing_values_columns[Missing_values_columns['% of Total Values']>25]
    Missing_values_columns=Missing_values_columns.index.tolist()   
    return Missing_values_columns

def cat_encoding(X):
    count = 0
    for col in X:
        if X[col].dtype == 'object':
            if len(list(X[col].unique())) <= 2:     
                le = preprocessing.LabelEncoder()
                X[col] = le.fit_transform(X[col])
                count += 1
                print (col)
            
    print('%d columns were label encoded.' % count)

    X= pd.get_dummies(X)
    return X

def corelations(X,threshold1,threshold2):
     # Create correlation matrix for against each varibale ###
    corr_matrix = X.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold1)]
    
    # Create correlation matrix for against target varibale ###
    corr = X.corr()['loan_status'].sort_values().abs()
    to_drop_wrt_target = [row for row in corr.index if corr[row]<threshold2 ]
    
    return to_drop,to_drop_wrt_target

def Prep_traning_test_data(training_data,test_data):
    ############### Training  data ####################
    
    ############################ filter the required class ##############################
    training_data = training_data[training_data['loan_status'] != 'Current']
    training_data['loan_status']=np.where(training_data['loan_status']=='Charged Off',-1,1)
    
    ############################## Removing High Missing Value Columns ###############################
    Missing_values_columns=missing_values(training_data)
    training_data.drop(Missing_values_columns, inplace=True, axis=1)
    
    ###################### Reemoving columns which doesn't make any sense to have in model #####################
    non_sense_columns=['id','member_id','emp_title','pymnt_plan','url','title','zip_code','initial_list_status',
                       'collections_12_mths_ex_med','policy_code','application_type','acc_now_delinq','chargeoff_within_12_mths',
                       'delinq_amnt','tax_liens','addr_state']
    #set(Missing_values_columns).intersection(non_sense_columns)
    training_data.drop(non_sense_columns, inplace=True, axis=1)
    
    ################################ Treating leftover missing values #################################
    #Missing_values_columns=null_values(training_data)
    training_data = training_data.dropna(axis=0) ## since no of missing values rows were less compares to whole data
    
    ############################### Special columns treatment ########################################
    special_cols=['int_rate','revol_util']
    training_data[special_cols]=training_data[special_cols].replace("%","",regex=True)
    training_data[special_cols]=training_data[special_cols].astype('float')
    
    ################################# Date columns #######################################
    date_cols=['last_pymnt_d','last_credit_pull_d','issue_d']
    training_data[date_cols]=training_data[date_cols].apply(pd.to_datetime, format='%d-%b')
    
    training_data['issue_month'] = pd.DatetimeIndex( training_data['last_credit_pull_d']).month
    training_data['credit_pull_month'] = pd.DatetimeIndex( training_data['last_credit_pull_d']).month
    training_data['last_payment_month'] = pd.DatetimeIndex( training_data['last_pymnt_d']).month
    
    training_data.drop(['last_pymnt_d','last_credit_pull_d','issue_d','earliest_cr_line'], inplace=True, axis=1)
    
    ################################### Encode Categorical Columns ##################################
    training_data=cat_encoding(training_data)
    
    ##################### checking correlations among each other ################################
    to_drop,to_drop_wrt_target=corelations(training_data,0.95,0.05)
    training_data.drop(to_drop+to_drop_wrt_target, axis=1, inplace=True)
    
    ################################## Normalization ##############################################
    scaler = preprocessing.StandardScaler()
    X=training_data.drop('loan_status',axis=1)
    scaler.fit(X)   
    X= scaler.transform(X)
    Y= training_data['loan_status'].values
    
    ############### Test  data ####################
    ############################ filter the required class ##############################
    test_data = test_data[test_data['loan_status'] != 'Current']
    test_data['loan_status']=np.where(test_data['loan_status']=='Charged Off',-1,1)
    
    ############ Dropping columns which were dropped in training data #####
    test_data.drop(non_sense_columns+Missing_values_columns, inplace=True, axis=1)
    
    ################################ Treating leftover missing values #################################
    test_data = test_data.dropna(axis=0) ## since no of missing values rows were less compares to whole data
    
    ############################### Special columns treatment ########################################
    special_cols=['int_rate','revol_util']
    test_data[special_cols]=test_data[special_cols].replace("%","",regex=True)
    test_data[special_cols]=test_data[special_cols].astype('float')
    
    ################################# Date columns #######################################
    date_cols=['last_pymnt_d','last_credit_pull_d','issue_d']
    test_data[date_cols]=test_data[date_cols].apply(pd.to_datetime, format='%d-%b')
    
    test_data['issue_month'] = pd.DatetimeIndex( test_data['last_credit_pull_d']).month
    test_data['credit_pull_month'] = pd.DatetimeIndex( test_data['last_credit_pull_d']).month
    test_data['last_payment_month'] = pd.DatetimeIndex( test_data['last_pymnt_d']).month
    
    test_data.drop(['last_pymnt_d','last_credit_pull_d','issue_d','earliest_cr_line'], inplace=True, axis=1)
    
    ################################### Encode Categorical Columns ##################################
    test_data=cat_encoding(test_data)
    
    ##################### checking correlations among each other ################################
    test_data.drop(to_drop+to_drop_wrt_target, axis=1, inplace=True)
    print('data preped')
    
    ################################## Normalization ##############################################
    X1=test_data.drop('loan_status',axis=1)
    X1= scaler.transform(X1)
    Y1= test_data['loan_status'].values
    
    return X,Y,X1,Y1
    
def run_gb( X,Y,X1,Y1,ntrees=100):
        
    ## Gradient Boositng
    model = GradientBoostingClassifier(n_estimators=ntrees)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    
    # report performance
    print('Mean % Accuracy: ',round(mean(n_scores)*100,2),'\nSTD of accuracy in %: ',round(std(n_scores)*100,2))
    
    ### modelling training data ###
    model.fit(X, Y)
    predict=model.predict(X1)
    accuracy=(predict== Y1)
    
    print("\nfinal test accuracy",round(sum(accuracy)/len(accuracy)*100,2))
    print('Precision: ',round((precision_score(Y1, predict))*100,2))
    print('Recall: ',round((recall_score(Y1, predict))*100,2))
    
def run_dt( X,Y,X1,Y1):
        
    ## Gradient Boositng
    model = DecisionTreeClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    
    # report performance
    print('Mean % Accuracy: ',round(mean(n_scores)*100,2),'\nSTD of accuracy in %: ',round(std(n_scores)*100,2))
    
    ### modelling training data ###
    model.fit(X, Y)
    predict=model.predict(X1)
    accuracy=(predict== Y1)
    
    precision_score(Y1,predict)
    print("\nfinal test accuracy",round(sum(accuracy)/len(accuracy)*100,2))
    print('Precision: ',round((precision_score(Y1, predict))*100,2))
    print('Recall: ',round((recall_score(Y1, predict))*100,2))
    

if __name__ == "__main__":
    
    # Load training data set
    training_data,test_data=load_data()
    
    #### Prepping data #####
    X,Y,X1,Y1=Prep_traning_test_data(training_data,test_data)
    
    ## Run Gradient Boosting    
    run_gb( X,Y,X1,Y1,100) ## 99.04%
    
    ## Run Decision Tree
    run_dt( X,Y,X1,Y1) ## 98.6%
    