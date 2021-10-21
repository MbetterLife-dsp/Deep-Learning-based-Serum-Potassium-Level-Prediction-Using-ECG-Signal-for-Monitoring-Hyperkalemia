
# -*- coding: utf-8 -*-

import sys
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from For_depth import DepthwiseConv1D # for modeling DCNN

np.random.seed(9)

#%% Build the DCRNN Model 
def build_model():
    # conv1
    #Separableconv1
    h = DepthwiseConv1D(128, padding='same')(X)
    h = keras.layers.Conv1D(16, 1,padding='same', activation='relu')(h)
    h = keras.layers.BatchNormalization(axis=-1)(h)
    h = keras.layers.MaxPool1D(pool_size=2)(h)

    #Separableconv2
    h = DepthwiseConv1D(64,padding='same')(h)
    h = keras.layers.Conv1D(16, 1,padding='same', activation='relu')(h)
    h = keras.layers.BatchNormalization(axis=-1)(h)
    h = keras.layers.MaxPool1D(pool_size=5)(h)
    
    #Separableconv3
    h = DepthwiseConv1D(64,padding='same')(h)
    h = keras.layers.Conv1D(16, 1,padding='same', activation='relu')(h)
    h = keras.layers.BatchNormalization(axis=-1)(h)

    #h=keras.layers.Flatten()(h)
    h = keras.layers.LSTM(10, return_sequences=False)(h)
    out = keras.layers.Dense(1)(h)
    model = keras.Model(inputs=X, outputs=out)

    
    # 4. 모델 학습과정 설정하기
    adam = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=adam,metrics=['mae'])
    model.summary()
    return model

#%% Load Datsets
dataset = np.loadtxt("Load the one-cycle ECGs datasets(.csv)", delimiter=",")
train_data = dataset[:, 0:270] #ECG length
train_targets = dataset[:, 274] #label position
X = keras.Input(shape=(270, 1))
epoch=500
k=5
kf=KFold(n_splits=k)
kf.get_n_splits(train_data)
num_val_samples=len(dataset)//k
estimated_value_whole_list=np.zeros((k,400))
val_targets_whole_list=np.zeros((k,400))
hist_list=np.zeros((k,epoch))
rotation=0
nn=0
#print(kf)

#%%  Partition datasets and train
#K-fold cross validation
for partial_train_data_index,val_data_index in kf.split(train_data): 
    print("%d -fold " % (rotation+1))
    #print("TRAIN:", partial_train_data_index, "TEST:", val_data_index)
    X_train,val_data=train_data[partial_train_data_index],train_data[val_data_index]
    Y_train,val_targets=train_targets[partial_train_data_index],train_targets[val_data_index]

    # dimension append
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    val_data = val_data.reshape((val_data.shape[0], val_data.shape[1], 1))
    model=build_model()
    hist=model.fit(X_train, Y_train, batch_size=32, epochs=epoch, verbose=2)
    estimated_value_list=[]
    hist_value=hist.history['loss']
    hist_list[rotation,0:len(hist_value)]=hist_value

#%% making the List for comparing target datasets and estimated datasets
    val_len=len(val_data)
    for n in range(val_len):
        test_x = val_data[n]
        test_x = test_x.reshape((1,)+test_x.shape)
        estimated_value=model.predict(test_x)
        estimated_value_list.append(estimated_value[0,0])
    val_targets_whole_list[rotation,0:len(val_targets)]=val_targets
    estimated_value_whole_list[rotation,0:len(estimated_value_list)]=estimated_value_list
    rotation=rotation+1


    non_list=[]
    for nn in range(k):
        nonzeros=np.argwhere(val_targets_whole_list[nn,:]==0)
        non_index=nonzeros[0,0]
        non_list.append(non_index)

    total_taget_list=[]
    total_estimated_list=[]
    for nn in range(k):
        for m in range(non_list[nn]):
            value=val_targets_whole_list[nn,m]
            total_taget_list.append(value)
            value=estimated_value_whole_list[nn,m]
            total_estimated_list.append(value)


#%% Code for making boxplots 
non_list=[]
for n in range(k):
    nonzeros=np.argwhere(val_targets_whole_list[n,:]==0)
    non_index=nonzeros[0,0]-1
    non_list.append(non_index)

#classfication estimated values
est_down_four=[]
est_four_five=[]
est_five_six=[]
est_six_seven=[]
est_seven_eight=[]
est_up_eight=[]
for m in range(len(non_list)):
    for n in range(non_list[m]):
        if val_targets_whole_list[m,n]<=4.0:
            est_down_four.append(estimated_value_whole_list[m,n])
        elif val_targets_whole_list[m,n]>4.0 and val_targets_whole_list[m,n]<=5.0:
            est_four_five.append(estimated_value_whole_list[m,n])
        elif val_targets_whole_list[m,n]>5.0 and val_targets_whole_list[m,n]<=6.0:
            est_five_six.append(estimated_value_whole_list[m,n])
        elif val_targets_whole_list[m,n]>6.0 and val_targets_whole_list[m,n]<=7.0:
            est_six_seven.append(estimated_value_whole_list[m,n])
        elif val_targets_whole_list[m,n]>7.0 and val_targets_whole_list[m,n]<=8.0:
            est_seven_eight.append(estimated_value_whole_list[m,n])
        elif val_targets_whole_list[m,n]>8.0:
            est_up_eight.append(estimated_value_whole_list[m,n])

fig, ax = plt.subplots()
ax.boxplot([est_down_four, est_four_five, est_five_six, est_six_seven ,est_seven_eight, est_up_eight])
plt.title('Multiple box plots of Hyperkalemia level')
plt.xticks([1, 2, 3, 4, 5, 6], ['<4.0', '4.0-5.0' ,'5.0-6.0','6.0-7.0','7.0-8.0','>=8.0'])
plt.yticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
plt.xlabel('Serum potassium level[mEq/L]')
plt.ylabel('Estimated pottasium level[mEq/L]')
plt.grid()
#image save
# fig=plt.gcf()
# fig.savefig('/dir/box_plot.pdf')
# plt.close()


#%% Code for making scatter plots 

"""
standard_x=np.arange(0,13)
standard_y=np.arange(0,13)
plt.figure(1)
plt.title("Scatter-plot of SPLs prediction")
plt.scatter(Total_target,Total_estimated,s = 10)
plt.plot(standard_x,standard_y)
plt.xlabel('SPLs[mEq/L]')
plt.ylabel('Estimated SPLs[mEq/L]')
plt.tight_layout()
plt.xticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0])
plt.yticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0])
plt.axis((0,12,0,12))
plt.grid()
fig=plt.gcf()
fig.savefig('/dir/scatter_plot.pdf')
plt.close()
"""

#%% calculation R-squared    

non_list=[]
for n in range(k):
    nonzeros=np.argwhere(val_targets_whole_list[n,:]==0)
    non_index=nonzeros[0,0]
    non_list.append(non_index)

total_taget_list=[]
total_estimated_list=[]
for n in range(k):
    for m in range(non_list[n]):
        value=val_targets_whole_list[n,m]
        total_taget_list.append(value)
        value=estimated_value_whole_list[n,m]
        total_estimated_list.append(value)
#t_i평균
sum=0
for n in range(len(total_taget_list)):
    a=total_taget_list[n]
    sum=sum+a
b=len(total_taget_list)
avg_t_i=sum/b

#오차
bbb=[]
for n in range(len(total_taget_list)):
    bb=total_taget_list[n]-total_estimated_list[n]
    bbb.append(bb)

#편차
vvv=[]
for n in range(len(total_taget_list)):
    vv=total_taget_list[n]-avg_t_i
    vvv.append(vv)


#오차 제곱
sum=0
for n in range(len(total_taget_list)):
    a=bbb[n]*bbb[n]
    sum=sum+a
sqr_bbb=sum

#편차 제곱

sum=0
for n in range(len(total_taget_list)):
    a=vvv[n]*vvv[n]
    sum=sum+a
sqr_vvv=sum

#R-squared

R_squared=1-(sqr_bbb/sqr_vvv)

print('R_squared:',R_squared)



