import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def count_accu(K,df):
    count = 0
    for i in range(K):
        if df.iloc[i,-1] == df.iloc[i,-2]:
            count+=1
    for i in range(K):
        if df.iloc[-(i+1),-1]== df.iloc[-(i+1),-2]:
            count+=1
    return count

def classification(y):
    if y >=0.5:
        return 1
    else:
        return 0
     
data = pd.read_csv('prediction/small_cap_predictions.csv',index_col = 0)
data['y_pred'] = data['y_prob'].apply(classification)
data = data.sort_values(['Date','y_prob'],ascending = True)
date_list = list(data.Date.unique())
data = data[['Date','Ticker','y_prob','y_pred','y_true']]
data = data.reset_index(drop = True)
# Overall accuracy of LSTM model
K = [2,5]
for k in K:
    accuracy_list = []
    for date in date_list:
        data_date = data[data.Date == date]
        data_date = data_date.reset_index(drop = True)
        accu_item = count_accu(k,data_date)
        accuracy_data = accu_item/(2*k)
        accuracy_list.append(accuracy_data)
    accuracy = round(sum(accuracy_list)/len(accuracy_list),4)
    print('Overall accuracy for small cap with '+str(k)+' stocks is '+str(accuracy))

y_pred = data['y_pred']
y_true = data['y_true']
y_prob = data['y_prob']
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_pred, y_true)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_pred, y_true)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_pred, y_true)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_pred, y_true)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(y_pred, y_true)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_pred, y_prob)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_pred, y_true)
print(matrix)