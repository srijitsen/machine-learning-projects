import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error
import math
from sklearn.neighbors import NearestNeighbors
from surprise import SVD
from surprise.model_selection import cross_validate

## Reading datasets###
tags = pd.read_csv('D:/documents/MDS IITH/data mining/Assignment 1/askubuntu/Tags.csv')
posts = pd.read_csv('D:/documents/MDS IITH/data mining/Assignment 1/askubuntu/Posts.csv')
posts=posts[['Id','OwnerUserId','ParentId','PostTypeId','Tags']]
#post_history = pd.read_csv('/content/drive/MyDrive/data_mining_data/PostHistory.csv')
#users = pd.read_csv('/content/drive/MyDrive/data_mining_data/Users.csv')

answer_post=posts[posts['PostTypeId'] == 2]
answer_post=answer_post[['OwnerUserId','ParentId']].drop_duplicates()

question_post=posts[posts['PostTypeId'] == 1]
question_post=question_post[['Id']].drop_duplicates()

answer_post=pd.merge(answer_post, question_post, left_on="ParentId",right_on="Id",how="inner")

answerer = answer_post.groupby(['OwnerUserId']).Id.agg(['nunique']).reset_index()
answerer = answerer.dropna(axis=0)
answerer['OwnerUserId']=answerer['OwnerUserId'].astype(int)

Tags = tags.groupby('Id').Count.agg(['sum']).reset_index()
Tags = Tags.dropna(axis=0)
Tags['Id']=Tags['Id'].astype(int)

answerer['rank']=answerer['nunique'].rank(axis=0, method='min',ascending=False) 
answerer[answerer['rank']<= 3].sort_values(by=['rank'])

Tags['rank']=Tags['sum'].rank(axis=0, method='min',ascending=False) 
Tags[Tags['rank']<= 3].sort_values(by=['rank'])

############ Question 2 ############
answer_post=posts[posts['PostTypeId'] == 2]
answer_post=answer_post[['OwnerUserId','ParentId']].drop_duplicates()

answerer=answerer[answerer['nunique'] >= 20]
Answer_tag=pd.merge(answer_post, answerer,how="inner")

question_post=posts[posts['PostTypeId'] == 1]
question_post=question_post[['Id','Tags']].drop_duplicates()

Answer_tag=pd.merge(Answer_tag, question_post, left_on="ParentId",right_on="Id",how="inner")
Answer_tag=Answer_tag[['OwnerUserId','ParentId','Tags']].drop_duplicates()

Tags=tags[tags['Count'] >= 20]
Tags=Tags[['Id','TagName','Count']].drop_duplicates()

for id in range(0,len(Tags)):
  value1=Tags.iat[id,1]
  value2=Tags.iat[id,0]  
  Answer_tag[value2]=np.where(Answer_tag['Tags'].str.contains(('<'+re.escape(value1)+'>'), case=False, na=False), 1, 0)
  print((id/len(Tags))*100)

Answer_tag=Answer_tag.drop(['ParentId','Tags'], axis=1)
Answer_tag_final=Answer_tag.groupby('OwnerUserId').sum().reset_index()
Answer_tag_final.to_csv('D:/documents/MDS IITH/data mining/Assignment 1/answer_tag_final_v2.csv')

#Answer_tag_final['answer_total']=Answer_tag_final.drop('OwnerUserId', axis=1).sum(axis=1)
#Answer_tag_final['OwnerUserId']=Answer_tag_final['OwnerUserId'].astype(str)
#Answer_tag_final.loc['total'] = Answer_tag_final.iloc[:, :-1].sum()
#Answer_tag_colsum = Answer_tag_final.sum(axis=0)

#Answer_tag_final=Answer_tag_final[Answer_tag_final.columns[Answer_tag_final.sum()>=20]]      
#Answer_tag_final=Answer_tag_final[Answer_tag_final['answer_total']>=20]
#Answer_tag_final=Answer_tag_final.sort_values(by=['OwnerUserId'])
#Answer_tag_final = Answer_tag_final.reindex(sorted(Answer_tag_final.columns), axis=1)

#Answer_tag_final.at['total', 'OwnerUserId'] = 'Tags_Total'

#Answer_tag_filtered1=Answer_tag_filtered
#Answer_tag_filtered1=Answer_tag_filtered1[['OwnerUserId']]
#Answer_tag=pd.read_csv('D:/documents/MDS IITH/data mining/Assignment 1/asnwer_tag.csv')
#Answer_tag=Answer_tag[['OwnerUserId','ParentId','175']]
#Answer_tag=pd.merge(Answer_tag_filtered1,Answer_tag,left_on='OwnerUserId',right_on='OwnerUserId', how="left")

# a=Answer_tag_filtered.head(100)

# Answer_tag_final[['175']].sum()
# a=Answer_tag[Answer_tag[175]==1]
# len(pd.unique(a['ParentId']))
# b=Answer_tag.head(10)

# a=Answer_tag_final[Answer_tag_final[175]==1]
# len(pd.unique(a['ParentId']))
# b=Answer_tag_final[['OwnerUserId','175']]
# sum(Answer_tag_final[175])

# Answer_tag_final1=Answer_tag_final
# Answer_tag_final1['sum']=Answer_tag_final1.drop('OwnerUserId', axis=1).sum(axis=1)
# Answer_tag_final1=Answer_tag_final1[['OwnerUserId','sum']]
# Answer_tag_final1=Answer_tag_final1[Answer_tag_final1['sum']>=20]
# d=Answer_tag_final.sum(axis=0)

###################################3 Question 3#############################################
#Tags['generic']=np.where(Tags['Count']>=250, 1,0)
generic_tags=Tags[Tags['Count']>= 250]['Id'].tolist()

Answer_tag_final1=Answer_tag_final.copy()

for col in Answer_tag_final1.columns[1:]:
    Answer_tag_final1['test']=col in generic_tags
    conditions  = [(Answer_tag_final1['test']==True) & (Answer_tag_final1[col]>=10),(Answer_tag_final1['test']!=True) & (Answer_tag_final1[col]>=3), 
                   (Answer_tag_final1['test']==True) & (Answer_tag_final1[col]<10),(Answer_tag_final1['test']!=True) & (Answer_tag_final1[col]<3) ]
    choices     = [ 1, 1, 0,0]
    Answer_tag_final1[col]=np.select(conditions, choices, default=np.nan)
    
Answer_tag_final1.drop('test', inplace=True, axis=1)
Answer_tag_final1.to_csv('D:/documents/MDS IITH/data mining/Assignment 1/answer_tag_final1_v2.csv')

a=round((len(Answer_tag_final1)-1)*0.85)
b=round((len(Answer_tag_final1.columns)-2)*0.85)

Answer_tag_final1=Answer_tag_final1.set_index('OwnerUserId')
Answer_tag_final1.index=Answer_tag_final1.index.astype('int')

test_data=Answer_tag_final1.iloc[a:, b:]
train_data=Answer_tag_final1.copy()

train_data.iloc[a:,b:].values[:]=0

#1
Answer_tag_final1.to_numpy().sum()
# 11135

#2
print(Answer_tag_final1.sum(axis=1).idxmax())
# 15811

#3
print(Answer_tag_final1.sum(axis=0).idxmax())
# 140

#4
train_data.to_numpy().sum()
# 10904

#5
test_data.shape
# (354,316)

#6 
test_data.to_numpy().sum()
#231

########################## Question 4 ####################################

#n_users = train_data.OwnerUserId.unique().shape[0]
#n_items = train_data.shape[1]-1

def User_User_Prediction(train_data_matrix,test_data_matrix,a,b,number_neighbors=3,method='weighted'):
        knn = NearestNeighbors(metric='jaccard', algorithm='brute')
        knn.fit(train_data_matrix.values)
        distances, indices = knn.kneighbors(train_data_matrix.values, n_neighbors=number_neighbors)
        distances=distances[a:]
        indices=indices[a:]
        print(number_neighbors)
        
        for tags in range(0,len(test_data_matrix.columns)-1):
            tag_value=test_data_matrix.columns[tags] 
            tag_index = train_data_matrix.columns.tolist().index(tag_value)
            
            #m=0
            for m,t in list(enumerate(train_data_matrix.iloc[a:,b:].index)):
                sim_movies = indices[m].tolist()
                movie_distances = distances[m].tolist()
            
                if m in sim_movies:
                  id_movie = sim_movies.index(m)
                  sim_movies.remove(m)
                  movie_distances.pop(id_movie) 
                else:
                  sim_movies = sim_movies[:number_neighbors-1]
                  movie_distances = movie_distances[:number_neighbors-1]
                  
                movie_similarity = [1-x for x in movie_distances]
                movie_similarity_copy = movie_similarity.copy()
                nominator = 0
                neighbour=0
                #s=0
                for s in range(0, len(movie_similarity)):
                    if train_data_matrix.iloc[sim_movies[s], tag_index] == 0:
                        if len(movie_similarity_copy) == (number_neighbors - 1):
                            movie_similarity_copy.pop(s)
                        else:
                            movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))
                    else:
                        if method=='weighted':
                            nominator = nominator + movie_similarity[s]*train_data_matrix.iloc[sim_movies[s],tag_index]
                        else:
                            nominator = nominator + train_data_matrix.iloc[sim_movies[s],tag_index]
                            neighbour=neighbour+1
                
                if len(movie_similarity_copy) > 0:
                    if sum(movie_similarity_copy) > 0:
                        if method=='weighted':
                            predicted_r = nominator/sum(movie_similarity_copy)
                        else:
                            predicted_r = nominator/(neighbour)
                    else:
                        predicted_r = 0
                else:
                  predicted_r = 0
                
                test_data_matrix.iloc[m,tags] = predicted_r
        return test_data_matrix       
                    
def Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,number_neighbors=3,method='weighted'):
        train_data_matrix=train_data_matrix.T      
        test_data_matrix=test_data_matrix.T
        knn = NearestNeighbors(metric='jaccard', algorithm='brute')
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
                sim_movies = indices[m].tolist()
                movie_distances = distances[m].tolist()
            
                if m in sim_movies:
                  id_movie = sim_movies.index(m)
                  sim_movies.remove(m)
                  movie_distances.pop(id_movie) 
                else:
                  sim_movies = sim_movies[:number_neighbors-1]
                  movie_distances = movie_distances[:number_neighbors-1]
                  
                movie_similarity = [1-x for x in movie_distances]
                movie_similarity_copy = movie_similarity.copy()
                nominator = 0
                neighbour=0
                #s=0
                for s in range(0, len(movie_similarity)):
                    if train_data_matrix.iloc[sim_movies[s], user_index] == 0:
                        if len(movie_similarity_copy) == (number_neighbors - 1):
                            movie_similarity_copy.pop(s)
                        else:
                            movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))
                    else:
                        if method=='weighted':
                            nominator = nominator + movie_similarity[s]*train_data_matrix.iloc[sim_movies[s],user_index]
                        else:
                            nominator = nominator + train_data_matrix.iloc[sim_movies[s],user_index]
                            neighbour=neighbour+1
                
                if len(movie_similarity_copy) > 0:
                    if sum(movie_similarity_copy) > 0:
                        if method=='weighted':
                            predicted_r = nominator/sum(movie_similarity_copy)
                        else:
                            predicted_r = nominator/(neighbour)
                    else:
                        predicted_r = 0
                else:
                  predicted_r = 0
                
                test_data_matrix.iloc[m,users] = predicted_r
        return test_data_matrix      

def RMSE_Calc(test_data,Prediction):
    mse = mean_squared_error(test_data, Prediction) 
    RMSE = math.sqrt(mse)
    print("Root Mean Square Error:\n",RMSE)  

train_data_matrix = train_data.copy()
#train_data_matrix=train_data_matrix.set_index('OwnerUserId')
#train_data_matrix.index=train_data_matrix.index.astype('int')

test_data_matrix = test_data.copy()
                  
Prediction_1=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,3,'weighted')           
Prediction_2=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,6,'weighted')           
Prediction_3=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,11,'weighted')           
                        
Prediction_4=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,3,'simple')           
Prediction_5=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,6,'simple')           
Prediction_6=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,11,'simple')           

RMSE_Calc(test_data,Prediction_1) # 0.046894550918141896
RMSE_Calc(test_data,Prediction_2) # 0.05456032447347553
RMSE_Calc(test_data,Prediction_3) # 0.08028279060828555
RMSE_Calc(test_data,Prediction_4) # 0.08028279060828555
RMSE_Calc(test_data,Prediction_5) # 0.08028279060828555
RMSE_Calc(test_data,Prediction_6) # 0.08028279060828555

Prediction_1=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,3,'weighted')           
Prediction_2=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,6,'weighted')           
Prediction_3=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,11,'weighted')           
                        
Prediction_4=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,3,'simple')           
Prediction_5=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,6,'simple')           
Prediction_6=Item_Item_Prediction(train_data_matrix,test_data_matrix,a,b,11,'simple')           

RMSE_Calc(test_data.T,Prediction_1) # 0.057121717874515185
RMSE_Calc(test_data.T,Prediction_2) # 0.057121717874515185
RMSE_Calc(test_data.T,Prediction_3) # 0.057121717874515185
RMSE_Calc(test_data.T,Prediction_4) # 0.057121717874515185
RMSE_Calc(test_data.T,Prediction_5) # 0.057121717874515185
RMSE_Calc(test_data.T,Prediction_6) # 0.057121717874515185

Prediction_1.to_csv('D:/documents/MDS IITH/data mining/Assignment 1/Prediction.csv')
    
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
        step
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

Pred1_1=run_matrix_factorization(a,b,R,N,M,2,0,0)
Pred1_2=run_matrix_factorization(a,b,R,N,M,3,0,0)
Pred1_3=run_matrix_factorization(a,b,R,N,M,5,0,0)

Pred2_1=run_matrix_factorization(a,b,R,N,M,2,0.001,0.003)
Pred2_2=run_matrix_factorization(a,b,R,N,M,3,0.001,0.003)
Pred2_3=run_matrix_factorization(a,b,R,N,M,5,0.001,0.003)

Pred3_1=run_matrix_factorization(a,b,R,N,M,2,0.05,0.05)
Pred3_2=run_matrix_factorization(a,b,R,N,M,3,0.05,0.05)
Pred3_3=run_matrix_factorization(a,b,R,N,M,5,0.05,0.05)

Pred4_1=run_matrix_factorization(a,b,R,N,M,2,0.5,0.75)
Pred4_2=run_matrix_factorization(a,b,R,N,M,3,0.5,0.75)
Pred4_3=run_matrix_factorization(a,b,R,N,M,5,0.5,0.75)

RMSE_Calc(np.array(test_data),Pred1_1)
RMSE_Calc(np.array(test_data),Pred1_2)
RMSE_Calc(np.array(test_data),Pred1_3)

RMSE_Calc(np.array(test_data),Pred2_1)
RMSE_Calc(np.array(test_data),Pred2_2)
RMSE_Calc(np.array(test_data),Pred2_3)

RMSE_Calc(np.array(test_data),Pred3_1)
RMSE_Calc(np.array(test_data),Pred3_2)
RMSE_Calc(np.array(test_data),Pred3_3)

RMSE_Calc(np.array(test_data),Pred4_1)
RMSE_Calc(np.array(test_data),Pred4_2)
RMSE_Calc(np.array(test_data),Pred4_3)


########################## Question 6 ########################
from surprise import accuracy, Dataset, SVD
from surprise.model_selection import train_test_split


trainset = train_data_matrix.copy()
testset = test_data_matrix.copy()

#algo = KNNBaseline(k=10)                  
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
predictions = algo.fit(trainset).test(testset)

# Then compute RMSE
accuracy.rmse(predictions
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)


Prediction_1=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,3,'weighted')           
Prediction_2=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,6,'weighted')           
Prediction_3=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,11,'weighted')           
                        
Prediction_4=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,3,'simple')           
Prediction_5=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,6,'simple')           
Prediction_6=User_User_Prediction(train_data_matrix,test_data_matrix,a,b,11,'simple')           

RMSE_Calc(test_data,Prediction_1) # 0.046894550918141896
RMSE_Calc(test_data,Prediction_2) # 0.05456032447347553
RMSE_Calc(test_data,Prediction_3) # 0.08028279060828555
RMSE_Calc(test_data,Prediction_4) # 0.08028279060828555
RMSE_Calc(test_data,Prediction_5) # 0.08028279060828555
RMSE_Calc(test_data,Prediction_6) # 0.08028279060828555