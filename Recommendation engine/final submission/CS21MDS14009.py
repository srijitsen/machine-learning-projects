######################## Libraries Used #############################
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error
import math
from sklearn.neighbors import NearestNeighbors
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise import Dataset
from surprise import KNNBaseline
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import Reader

########################### Reading datasets #####################################
tags = pd.read_csv('D:/documents/MDS IITH/data mining/Assignment 1/askubuntu/Tags.csv')
posts = pd.read_csv('D:/documents/MDS IITH/data mining/Assignment 1/askubuntu/Posts.csv')
posts=posts[['Id','OwnerUserId','ParentId','PostTypeId','Tags']]

##### Filtering for answers ###############
answer_post=posts[posts['PostTypeId'] == 2]
answer_post=answer_post[['OwnerUserId','ParentId']].drop_duplicates()

########### filtering for questions ###############
question_post=posts[posts['PostTypeId'] == 1]
question_post=question_post[['Id']].drop_duplicates()

############ Merging the question on answers ##################
answer_post=pd.merge(answer_post, question_post, left_on="ParentId",right_on="Id",how="inner")

############ Counting questions answered by the users ##########
answerer = answer_post.groupby(['OwnerUserId']).Id.agg(['nunique']).reset_index()
answerer = answerer.dropna(axis=0)
answerer['OwnerUserId']=answerer['OwnerUserId'].astype(int)

########### Counting tags annotated to questions #######
Tags = tags.groupby('Id').Count.agg(['sum']).reset_index()
Tags = Tags.dropna(axis=0)
Tags['Id']=Tags['Id'].astype(int)

######### Finding top 3 in each using rank ########
answerer['rank']=answerer['nunique'].rank(axis=0, method='min',ascending=False) 
answerer[answerer['rank']<= 3].sort_values(by=['rank'])

Tags['rank']=Tags['sum'].rank(axis=0, method='min',ascending=False) 
Tags[Tags['rank']<= 3].sort_values(by=['rank'])

######### Output to question 1 ##########
#    Id    sum  rank
# 118  175  22237   1.0
# 319  538  21225   2.0
# 98   140  21150   3.0

############################################## Question 2 #######################################

#### Creating answers table ######
answer_post=posts[posts['PostTypeId'] == 2]
answer_post=answer_post[['OwnerUserId','ParentId']].drop_duplicates()

######## Removing the users which had less than 20 answers #########
answerer=answerer[answerer['nunique'] >= 20]
Answer_tag=pd.merge(answer_post, answerer,how="inner")

########## Question table #########
question_post=posts[posts['PostTypeId'] == 1]
question_post=question_post[['Id','Tags']].drop_duplicates()

##### merging it back with answer to get the only answer which were actually on question list ####
Answer_tag=pd.merge(Answer_tag, question_post, left_on="ParentId",right_on="Id",how="inner")
Answer_tag=Answer_tag[['OwnerUserId','ParentId','Tags']].drop_duplicates()

######### Removing tags which were annonated less than 20 times ##########
Tags=tags[tags['Count'] >= 20]
Tags=Tags[['Id','TagName','Count']].drop_duplicates()

########## Creating individual tag columns from question table to create utility matrix (Takes some time to run) #############
for id in range(0,len(Tags)):
  value1=Tags.iat[id,1]
  value2=Tags.iat[id,0]  
  Answer_tag[value2]=np.where(Answer_tag['Tags'].str.contains(('<'+re.escape(value1)+'>'), case=False, na=False), 1, 0)
  print((id/len(Tags))*100)

#### Final touches to create utility matrix ##########
Answer_tag=Answer_tag.drop(['ParentId','Tags'], axis=1)
Answer_tag_final=Answer_tag.groupby('OwnerUserId').sum().reset_index()
#Answer_tag_final.to_csv('D:/documents/MDS IITH/data mining/Assignment 1/answer_tag_final_v2.csv')

###################################3 Question 3#############################################
  
############# Creating ratings ################
Answer_tag_final1=Answer_tag_final.copy()

for col in Answer_tag_final1.columns[1:]:
    Answer_tag_final1[col]=np.where(Answer_tag_final1[col]>=15, 5,(Answer_tag_final1[col]/3))
    Answer_tag_final1[col] = Answer_tag_final1[col].apply(np.floor)
    
Answer_tag_final1=Answer_tag_final1.loc[(Answer_tag_final1.sum(axis=1) != 0), (Answer_tag_final1.sum(axis=0) != 0)]    

#### Getting indexes of bottom and left 15% #########
a=round((len(Answer_tag_final1)-1)*0.85)
b=round((len(Answer_tag_final1.columns)-2)*0.85)

Answer_tag_final1=Answer_tag_final1.set_index('OwnerUserId')
Answer_tag_final1.index=Answer_tag_final1.index.astype('int')

##### splitting test and train and replacing test cases with 0 value #############
test_data=Answer_tag_final1.iloc[a:, b:]
train_data=Answer_tag_final1.copy()
train_data.iloc[a:,b:].values[:]=0

############# Answers to question 3 ################

#1 total sum
Answer_tag_final1.to_numpy().sum()
# 822257

#2 ID of max row sum
print(Answer_tag_final1.sum(axis=1).idxmax())
# 15811 (User id)

#3 Id of max column sum
print(Answer_tag_final1.sum(axis=0).idxmax())
# 140 (Tag id)

#4 Total sum of train 
train_data.to_numpy().sum()
# 80380

#5 Dimensions of Test
test_data.shape
# (354,235)

#6 total sum of Test
test_data.to_numpy().sum()
#1877

########################## Question 4 ####################################

def User_User_Prediction(train_data_matrix,test_data_matrix,a,b,number_neighbors=3,method='weighted',distance_metric= 'correlation'):
        knn = NearestNeighbors(metric=distance_metric, algorithm='brute')
        knn.fit(train_data_matrix.values)
        distances, indices = knn.kneighbors(train_data_matrix.values, n_neighbors=number_neighbors)
        distances=distances[a:]
        indices=indices[a:]
        print(number_neighbors)
        
        for tags in range(0,len(test_data_matrix.columns)-1):
            tag_value=test_data_matrix.columns[tags] 
            tag_index = train_data_matrix.columns.tolist().index(tag_value)
            
            for m,t in list(enumerate(train_data_matrix.iloc[a:,b:].index)):
                sim_tags = indices[m].tolist()
                tag_distances = distances[m].tolist()
            
                if m in sim_tags:
                  id_movie = sim_tags.index(m)
                  sim_tags.remove(m)
                  tag_distances.pop(id_movie) 
                else:
                  sim_tags = sim_tags[:number_neighbors-1]
                  tag_distances = tag_distances[:number_neighbors-1]
                  
                tag_similarity = [1-x for x in tag_distances]
                tag_similarity_copy = tag_similarity.copy()
                nominator = 0
                neighbour=0
                #s=0
                for s in range(0, len(tag_similarity)):
                    if train_data_matrix.iloc[sim_tags[s], tag_index] == 0:
                        if len(tag_similarity_copy) == (number_neighbors - 1):
                            tag_similarity_copy.pop(s)
                        else:
                            tag_similarity_copy.pop(s-(len(tag_similarity)-len(tag_similarity_copy)))
                    else:
                        if method=='weighted':
                            nominator = nominator + tag_similarity[s]*train_data_matrix.iloc[sim_tags[s],tag_index]
                        else:
                            nominator = nominator + train_data_matrix.iloc[sim_tags[s],tag_index]
                            neighbour=neighbour+1
                
                if len(tag_similarity_copy) > 0:
                    if sum(tag_similarity_copy) > 0:
                        if method=='weighted':
                            predicted_r = nominator/sum(tag_similarity_copy)
                        else:
                            predicted_r = nominator/(neighbour)
                    else:
                        predicted_r = 0
                else:
                  predicted_r = 0
                
                test_data_matrix.iloc[m,tags] = predicted_r
        return test_data_matrix       
                    
def Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,number_neighbors=3,method='weighted',distance_metric='correlation'):
        train_data_matrix=train_data_matrix.T      
        test_data_matrix=test_data_matrix.T
        knn = NearestNeighbors(metric=distance_metric, algorithm='brute')
        knn.fit(train_data_matrix.values)
        distances, indices = knn.kneighbors(train_data_matrix.values, n_neighbors=number_neighbors)
        distances=distances[b:]
        indices=indices[b:]
        print(number_neighbors)
        
        for users in range(0,len(test_data_matrix.columns)-1):
            user_value=test_data_matrix.columns[users] 
            user_index = train_data_matrix.columns.tolist().index(user_value)
            
            #m=0
            for m,t in list(enumerate(train_data_matrix.iloc[b:,a:].index)):
                sim_users = indices[m].tolist()
                user_distances = distances[m].tolist()
            
                if m in sim_users:
                  id_movie = sim_users.index(m)
                  sim_users.remove(m)
                  user_distances.pop(id_movie) 
                else:
                  sim_users = sim_users[:number_neighbors-1]
                  user_distances = user_distances[:number_neighbors-1]
                  
                user_similarity = [1-x for x in user_distances]
                user_similarity_copy = user_similarity.copy()
                nominator = 0
                neighbour=0
                #s=0
                for s in range(0, len(user_similarity)):
                    if train_data_matrix.iloc[sim_users[s], user_index] == 0:
                        if len(user_similarity_copy) == (number_neighbors - 1):
                            user_similarity_copy.pop(s)
                        else:
                            user_similarity_copy.pop(s-(len(user_similarity)-len(user_similarity_copy)))
                    else:
                        if method=='weighted':
                            nominator = nominator + user_similarity[s]*train_data_matrix.iloc[sim_users[s],user_index]
                        else:
                            nominator = nominator + train_data_matrix.iloc[sim_users[s],user_index]
                            neighbour=neighbour+1
                
                if len(user_similarity_copy) > 0:
                    if sum(user_similarity_copy) > 0:
                        if method=='weighted':
                            predicted_r = nominator/sum(user_similarity_copy)
                        else:
                            predicted_r = nominator/(neighbour)
                    else:
                        predicted_r = 0
                else:
                  predicted_r = 0
                
                test_data_matrix.iloc[m,users] = predicted_r
        return test_data_matrix      

def RMSE_Calc(test_data,Prediction,type):
    if type=="user":
        test_data=test_data.reset_index()
        test_data=test_data.melt(id_vars=["OwnerUserId"],var_name="Tags",value_name="Value")
        test_data=test_data[test_data['Value'] >0]
        
        Prediction=Prediction.reset_index()
        Prediction=Prediction.melt(id_vars=["OwnerUserId"],var_name="Tags",value_name="Pred")
        
        test_data=pd.merge(test_data, Prediction, left_on=["OwnerUserId","Tags"],right_on=["OwnerUserId","Tags"],how="inner")
    else:
        test_data=test_data.T
        test_data=test_data.reset_index()
        test_data=test_data.melt(id_vars=["index"],var_name="user",value_name="Value")
        test_data=test_data[test_data['Value'] >0]
        
        Prediction=Prediction.reset_index()
        Prediction=Prediction.melt(id_vars=["index"],var_name="user",value_name="Pred")
        
        test_data=pd.merge(test_data, Prediction, left_on=["index","user"],right_on=["index","user"],how="inner")
        
    mse = mean_squared_error(test_data[["Value"]], test_data[["Pred"]]) 
    RMSE = math.sqrt(mse)
    print("Root Mean Square Error:\n",RMSE)  

train_data_matrix = train_data.copy()
test_data_matrix = test_data.copy()

## User - User Prediction and RMSE ######
Prediction_1=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,3,'weighted','correlation')     
RMSE_Calc(test_data,Prediction_1,"user") # 2.2027173550466173
      
Prediction_2=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,4,'weighted','correlation')  
RMSE_Calc(test_data,Prediction_2,"user") #  2.1556209381035507
         
Prediction_3=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,6,'weighted','correlation')           
RMSE_Calc(test_data,Prediction_3,"user") #   2.0709038124843726
                        
Prediction_4=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,3,'simple','correlation') 
RMSE_Calc(test_data,Prediction_4,"user") # 2.2027173550466173
          
Prediction_5=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,4,'simple','correlation')    
RMSE_Calc(test_data,Prediction_5,"user") # 2.1557553123597706

Prediction_6=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,6,'simple','correlation')           
RMSE_Calc(test_data,Prediction_6,"user") #  2.070477285284168

## Item-Item Prediction and RMSE ######
Prediction_1=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,3,'weighted','correlation')   
RMSE_Calc(test_data,Prediction_1,"item") #  2.227268523444962
        
Prediction_2=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,4,'weighted','correlation')      
RMSE_Calc(test_data,Prediction_2,"item") #2.215646015530855
     
Prediction_3=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,6,'weighted','correlation')           
RMSE_Calc(test_data,Prediction_3,"item") # 2.1053920070768015
                        
Prediction_4=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,3,'simple','correlation')   
RMSE_Calc(test_data,Prediction_4,"item") # 2.227268523444962
        
Prediction_5=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,4,'simple','correlation')           
RMSE_Calc(test_data,Prediction_5,"item") # 2.2156520025888615

Prediction_6=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,6,'simple','correlation')           
RMSE_Calc(test_data,Prediction_6,"item") # 2.105692370765463

#Prediction_1.to_csv('D:/documents/MDS IITH/data mining/Assignment 1/Prediction.csv')
    
############################ Question 4 ###################################    

def matrix_factorization(R, P, Q, K, steps=100, alpha=0.0005, beta1=0.0,beta2=0.0):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T

    for step in range(steps):
        #print(step)
        for i in range(len(R)):            
            for j in range(len(R[i])):                
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        if beta1!=0.0 and beta2!=0.0:
                        # calculate gradient with a and beta parameter
                           P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta1 * P[i][k])
                           Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta2 * Q[k][j])
                        else:
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k])

        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    
                    if beta1!=0 and beta2!=0:
                        for k in range(K):
                            e = e + (beta1)*pow(P[i][k],2) + (beta2)*pow(Q[k][j],2)
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T

R = np.array(train_data_matrix)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])

def run_matrix_factorization(a,b,R,N,M,K,beta1,beta2):
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    nP, nQ = matrix_factorization(R, P, Q, K,100,0.005,beta1,beta2)
    nR = np.dot(nP, nQ.T)
    Pred=nR[a:,b:]
    return Pred

def RMSE_Calc(test_data,Prediction):
    test_data=pd.DataFrame(test_data)
    test_data=test_data.reset_index()
    test_data=test_data.melt(id_vars=["index"],var_name="Tags",value_name="Value")
    test_data=test_data[test_data['Value'] >0]
    
    Prediction=pd.DataFrame(Prediction)    
    Prediction=Prediction.reset_index()
    Prediction=Prediction.melt(id_vars=["index"],var_name="Tags",value_name="Pred")
        
    test_data=pd.merge(test_data, Prediction, left_on=["index","Tags"],right_on=["index","Tags"],how="inner")        
    mse = mean_squared_error(test_data[["Value"]], test_data[["Pred"]]) 
    RMSE = math.sqrt(mse)
    print("Root Mean Square Error:\n",RMSE)  
    
Pred1_1=run_matrix_factorization(a,b,R,N,M,2,0,0)
RMSE_Calc(np.array(test_data),Pred1_1) # 1.3355263093080478

Pred1_2=run_matrix_factorization(a,b,R,N,M,5,0,0)
RMSE_Calc(np.array(test_data),Pred1_2) # 1.5939510087134594

Pred1_3=run_matrix_factorization(a,b,R,N,M,10,0,0)
RMSE_Calc(np.array(test_data),Pred1_3) # 1.7552190926828364

Pred2_1=run_matrix_factorization(a,b,R,N,M,2,0.001,0.003)
RMSE_Calc(np.array(test_data),Pred2_1) #  1.372190107756691

Pred2_2=run_matrix_factorization(a,b,R,N,M,5,0.001,0.003)
RMSE_Calc(np.array(test_data),Pred2_2) # 1.602625562513239

Pred2_3=run_matrix_factorization(a,b,R,N,M,10,0.001,0.003)
RMSE_Calc(np.array(test_data),Pred2_3) # 1.661273809571962

Pred3_1=run_matrix_factorization(a,b,R,N,M,2,0.05,0.05)
RMSE_Calc(np.array(test_data),Pred3_1) # 1.2701829630061812

Pred3_2=run_matrix_factorization(a,b,R,N,M,3,0.05,0.05)
RMSE_Calc(np.array(test_data),Pred3_2) # 1.2745416799699423

Pred3_3=run_matrix_factorization(a,b,R,N,M,5,0.05,0.05)
RMSE_Calc(np.array(test_data),Pred3_3) # 1.3178495429289911

Pred4_1=run_matrix_factorization(a,b,R,N,M,2,0.5,0.75)
RMSE_Calc(np.array(test_data),Pred4_1) # 1.2586525163333495

Pred4_2=run_matrix_factorization(a,b,R,N,M,3,0.5,0.75)
RMSE_Calc(np.array(test_data),Pred4_2) # 1.2519551062109762

Pred4_3=run_matrix_factorization(a,b,R,N,M,5,0.5,0.75)
RMSE_Calc(np.array(test_data),Pred4_3) #  1.252905479093929

########################## Question 6 ########################

### Rating scale #####
reader = Reader(rating_scale=(1, 5))

### Prepping data for surprise library ###

## Train Data ##
train_data2=train_data.copy()
train_data2=train_data2.reset_index()
train_data2=train_data2.melt(id_vars=["OwnerUserId"],var_name="Tags",value_name="Value")
train_data2=train_data2[train_data2['Value'] >0]
data = Dataset.load_from_df( train_data2, reader)

## Test Data ##
test_data2=test_data.copy()
test_data2=test_data2.reset_index()
test_data2=test_data2.melt(id_vars=["OwnerUserId"],var_name="Tags",value_name="Value")
test_data2=test_data2[test_data2['Value'] >0]
test_data2 = test_data2.values.tolist()

########################  KNN Baseline ############################################
def surprise_prediction_run(k,user,data,test_data):
    sim_options = { 'name': 'pearson' ,'user_based':  user}
    kf = KFold(n_splits=10)
    algo = KNNBaseline(k =3 , sim_options = sim_options)
    best_algo = None
    best_rmse = 1000.0
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)
        # Compute and print Root Mean Squared Error
        rmse = accuracy.rmse(predictions, verbose=True)
        if rmse < best_rmse:
            best_rmse= rmse
            best_algo = algo
    pred = best_algo.test(test_data)
    pred=pd.DataFrame(pred)        
    return pred        

##User-User##
pred=surprise_prediction_run(3,True,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) #1.4997100298806927

pred=surprise_prediction_run(4,True,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) #1.4713929565149513

pred=surprise_prediction_run(6,True,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) #1.494432826750964

## Item-Item ###
pred=surprise_prediction_run(3,False,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) #1.3237088291051906

pred=surprise_prediction_run(4,False,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) #1.3046869978360067

pred=surprise_prediction_run(6,False,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) #1.3228792789149826

#################################### SVD ####################################

## Grid search to find best parameters for SVD #####
param_grid = {"n_epochs": [5, 10], "lr_all": [0.0005, 0.005], "reg_all": [0.001, 0.75]}
gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
gs.fit(data)
print(gs.best_params["rmse"])
#{'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.001}

def svd_prediction_run(k,data,test_data):
    kf = KFold(n_splits=10)
    algo = SVD(n_factors=k, n_epochs=10,lr_all=0.005,reg_all=0.001)
    best_algo = None
    best_rmse = 1000.0
    for trainset, testset in kf.split(data):
        # train and test algorithm.        
        algo.fit(trainset)
        predictions = algo.test(testset)
        # Compute and print Root Mean Squared Error
        rmse = accuracy.rmse(predictions, verbose=True)
        if rmse < best_rmse:
            best_rmse= rmse
            best_algo = algo    
    pred = best_algo.test(test_data)
    pred=pd.DataFrame(pred)        
    return pred        

def RMSE_Calc(test_data,Prediction):
    mse = mean_squared_error(test_data, Prediction) 
    RMSE = math.sqrt(mse)
    print("Root Mean Square Error:\n",RMSE)  
    
pred=svd_prediction_run(2,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) # 1.2463063784810309

pred=svd_prediction_run(5,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) # 1.2364574614799242

pred=svd_prediction_run(10,data,test_data2)
RMSE_Calc(pred[['r_ui']],pred[['est']]) # 1.2302573185954417


