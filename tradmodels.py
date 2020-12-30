'''
Contains methods that define the traditional models
'''
import pandas as pd
import numpy as np
import helper as h

def scale_data(pk_df):
  #scale data((X-mean)/std_dev)
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X = sc.fit_transform(pk_df)
  return X

def set_split(target_set,noise_set):
  #get the size of the noise_set for use in the model
  size=int((target_set.shape[0]/noise_set.shape[0])*1.2*noise_set.shape[0])

  #randomly select 1.2x the target_set rows from the noise set
  from numpy.random import default_rng
  msk = default_rng(seed=1).uniform(0,1,len(noise_set)) < size
  noise_set = noise_set[msk]
  noise_set.label=0
  target_set.label=1
  return h.splitdata(noise_set.append(target_set),dev_size=0.2,r_state=1)

def train_model(X_train,y_train):
  #create model
  from sklearn.ensemble import RandomForestClassifier
  return RandomForestClassifier(n_estimators=100).fit(X_train,y_train)

def assess_model(model,X_dev,y_dev,m_label,thresh=0.0,id_val='0'):
  #check model
  print(model.score(X_dev,y_dev))

  preds=model.predict_proba(X_dev)

  from sklearn.metrics import confusion_matrix
  cm=confusion_matrix(y_dev,h.dec_pred(preds,thresh))

  print('Model label ',m_label,'\n',cm,'\n\n\n')
  print('Model prectictions',model.predict_proba(X_dev))
  output_probs=pd.DataFrame(preds,index=y_dev.index.values)
  output_probs['label']=y_dev
  output_probs.to_csv('Model Data/Binary Model/Single Models/Single Models'+str(m_label)+id_val+'_probs.csv',mode='w')
  pd.DataFrame(cm).to_csv('Model Data/Binary Model/Single Models/Single Models'+str(m_label)+id_val+'_confmat.csv',mode='w')

def run_pca(df):
  #print(df)
  temp_df=df.drop(['label'],axis=1)

  from sklearn.decomposition import PCA
  pca=PCA(0.8)
  output_df=pd.DataFrame(pca.fit_transform(temp_df.values),index=df.index.values)
  output_df['label']=df['label']
  #print(output_df)
  return output_df,pca

def train_models(fin_path):
  """Trains a binary classifier model for each class in the training set and
  returns a list of those binary_classifier_probs

  Args:
    fin_path: string, path to the folder with the intended filepaths

  Returns:
    a tuple with the following: list of scikit-learn random forest models, list
    of labels included in the training set, and a list of PCA models for each
    label
  """
  #set filepaths for import/export
  fin_path=r'Data/Preprocessed/Continuous Wavelet Transformation/Labeled/'

  #build master dataframe of data from fin_path and run pca to reduce the feature set
  df,pca=run_pca(h.dfbuilder(fin_path,synth=False,split_df=False,dev_size=.2,r_state=1,directory=directory,use_trash=False))

  #list for holding models
  m_list=[]

  #loop to create one model for each label in df
  l_list=sorted(df['label'].unique().tolist())
  for i in l_list:
    print('in loop, label is',i,'\n\n')
    #for label i, select the target label set and create a training and dev set
    target_df=df.loc[df['label']==i,:]
    noise_df=df.loc[df['label']!=i,:]
    X_train,X_dev,y_train,y_dev=set_split(target_df,noise_df)
    #print('\ndata split\n',X_train.head(),'\n\n',y_train.head())
    m_list.append(train_model(X_train,y_train.values.ravel()))
    print('model trained')
    print(y_train)
    assess_model(m_list[-1],X_train,y_train,i,id_val='train')
    assess_model(m_list[-1],X_dev,y_dev,i,id_val='dev')
  return m_list,l_list,pca

def model_set(fin_path=r'Data/CWT Data/Single/',testin_path=r'/Data/CWT Data/Single/',raw=False):
  """Trains a series of binary models, one for each label in the data set found
  in fin_path and tests against data in the testin_path

  Args:
    fin_path: string, path to folder with training data files
    testin_path: string, path to folder with the test data
    raw: boolean, True if the input is raw data, false if it has been Preprocessed

  Returns:
    Tuple of objects in the following order: DataFrame of predicted labels, a
    list of trained binary models, a list of trained PCA models
"""

  #train the models for testing
  m_list,l_list,pca_list=train_models(fin_path)


  '''should this and the following section be combined into one method?'''
  init_test_df=h.dfbuilder(testin_path,split_df=False,r_state=1,raw=raw)
  print(init_test_df)

  '''double check test process'''
  test_df=pd.DataFrame(pca.transform(init_test_df.drop(['label'],axis=1).values),index=init_test_df.index.values)
  test_df['label']=init_test_df['label']
  print(len(test_df.index),",",len(m_list))


  labels=np.empty(shape=(len(test_df.index),len(m_list)))


  for j in range(len(m_list)):
    labels[:,j]=m_list[j].predict_proba(test_df.drop(['label'],axis=1))[:,1]
  label_df=pd.DataFrame(labels,index=test_df.index.values,columns=l_list)
  label_df['label']=test_df['label']


  '''Save Model Data - ensure filepath exists'''
  label_df.to_csv('Model Data/Binary Model/binary_classifier_probs.csv',mode='w')

  return label_df,m_list,pca
