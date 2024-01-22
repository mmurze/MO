import itertools

from sklearn.metrics import classification_report, confusion_matrix
from imagePreprocessing import result
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
        
# pics=["pic1.jpg", "pic2.jpg", "pic4.jpg", "pic5.jpg", "pic6.jpg", "pic7.jpg"]
  
def plot_pred_test(pred, test, name_pred, name_test):
  plt.figure(figsize=(10, 5))
  plt.plot(pred, "r--o" , label=name_pred, markersize=12)
  plt.plot(test,"b--o", label=name_test, markersize=7)
  plt.ylabel('ответ на поставленный вопрос (0 - нет, 1 -да)')
  plt.savefig("pred_test.png")
  plt.show()
  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.savefig("conf_matrix.png")
    plt.show()
  
if __name__ == "__main__":
  train = pd.read_csv('dataset.csv')   
  
  X = train.drop(["drink"], axis=1).values
  y = train["drink"].values
  
  
  X_tarin, X_test, y_train, y_test = train_test_split(X, y)        

  model = GradientBoostingClassifier()
  model.fit(X_tarin, y_train)

  predict = model.predict(X_test)
  plot_pred_test(predict, y_test, "pred", "test")
  
  font = {'size' : 10}
  plt.rc('font', **font)

  cnf_matrix = confusion_matrix(y_test, predict)

  plot_confusion_matrix(cnf_matrix, classes=['Нет', 'Да'],
                        title='Матрица Ошибок')
  
  
  report = classification_report(y_test, predict, target_names=['Нет', 'Да'])
  print(report)

  person = [158, 50, 18, 1, 0]
  milk = result("pic1.jpg")
  line = np.array([milk + person])
  print(model.predict(line))
