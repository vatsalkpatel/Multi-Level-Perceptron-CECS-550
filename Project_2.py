
# Multi Level Perceptron Model for Given Dataset
# Created by Sean, Sushmita, Vatsal
# Libraries Used :-
import re
import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns 

# Function to read both input files to create Training Set, Validation Set, Test Set
def read_input_file(file_name,k):

    feature_vectors = []
    file = open(file_name,'r')

    for line in file:
        vectors = []
        data = re.split(r'[()\s]\s*', line)
        while '' in data:
            data.remove('')

        for item in data:
            vectors.append(int(item))
        feature_vectors.append(vectors)
    if k==1:
        with open("input.csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(feature_vectors)
    else:
        with open("input_2.csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(feature_vectors)        
read_input_file('ClassifiedSetData.txt',1)
read_input_file('TestSetData.txt',2)


# Class for creating MLP
# Functions :-
# 1) init - To initialize required data
# 2) split_data - Splits Data into train, validate and test
# 3) multiplication_between_layers - calculates total input to a perceptron node by $\sum_{i=1}^{n} x_i*w_i$
# 4) merge_data - Merge train and validate sets
# 5) logistic - Activation Function
# 6) forward_propagation - to classify based on inputs
# 7) backward_propagation - to Update Weight as per the result
# 8) accuracy - calculates accuracy of the model
# 9) confusion_matrix - Generates Confusion Matrix
# 10) confusion_matrix_create - preps matrix for beautiful display
class MLP:
    def __init__(self):
        self.number_of_nodes=10
        self.max_feature_value=96
        self.num_output_nodes=8
        self.True_Positive=[0]*8
        self.True_Negative=[0]*8
        self.False_Positive=[0]*8
        self.False_Negative=[0]*8
        
    def split_data(self):
        self.df=pd.read_csv('input.csv',header=None)
        self.df1=pd.read_csv('input_2.csv',header=None)
        self.df=self.df.rename({0:'ID',1:'Feature_1',2:'Feature_2',3:'Feature_3',4:'Feature_4',5:'Feature_5',6:'Feature_6',7:'Feature_7',8:'Feature_8',9:'Feature_9',10:'Feature_10',11:'Output'}, axis=1)
        self.df1=self.df1.rename({0:'ID',1:'Feature_1',2:'Feature_2',3:'Feature_3',4:'Feature_4',5:'Feature_5',6:'Feature_6',7:'Feature_7',8:'Feature_8',9:'Feature_9',10:'Feature_10',11:'Output'}, axis=1)
        self.train_data, self.validate_data, self.test_data = np.split(self.df.sample(frac=1,random_state=42), [int(.6*len(self.df)), int(.8*len(self.df))])
    def multiplication_between_layers(self,weights, inputs):
        total=0
        for i in range(len(weights)):
            total += (inputs[i] * weights[i])
        return total   
    def merge_data(self):
        frames = [self.train_data,self.validate_data]
        self.train_data=pd.concat(frames) 
    def logistic(self, x):
        return 1 / ( 1 + math.exp(-x))

    def normalize_array(self, features,normalize_value):
        normalized_array=[]
        for i in features:
            normalized_array.append(i/normalize_value)
        return normalized_array
    
    def initialize_weight(self,x_dim,y_dim):
        return np.random.randn(x_dim,y_dim)*np.sqrt(1/y_dim)
    
    def set_weights(self,Wh1,Wh2):
        print("Initial Weight :")
        self.Wh1=Wh1
        print("Hidden Layer Weight = ",self.Wh1)
        self.Wh2=Wh2
        print("Output Layer Weight = ",self.Wh2)
        
    def forward_propagation(self,training_data):
            

        self.hidden_layer = []
        for i in range(self.number_of_nodes):
            self.hidden_layer.append(self.logistic(self.multiplication_between_layers(self.Wh1[i], training_data)))

        self.output_layer = []
    
        for i in range(self.num_output_nodes):
            self.output_layer.append(self.logistic(self.multiplication_between_layers(self.Wh2[i], self.hidden_layer)))
        return self.hidden_layer, self.output_layer
 
    def back_propagation(self,learning_rate, min_accuracy):
        epochs = 0
        while self.accuracy(self.train_data) < min_accuracy:
            epochs += 1
            for value in self.train_data.values:
                training_vector = self.normalize_array(list(value)[1:11], self.max_feature_value)
                self.forward_propagation(training_vector)
                error_1= [] 
                for j in range(self.num_output_nodes):
                    p = self.output_layer[j]
                    learning_rate = 0.3
                    if j == list(value)[-1]:
                        learning_rate = 0.7
                    error_1.append(p * (1 - p) * (learning_rate - p))
  
                error_2 = [] 
                for j in range(self.number_of_nodes):
                    hidden_layer_input = self.hidden_layer[j]
                    output_with_error = self.multiplication_between_layers(self.Wh2[:,j], error_1)
                    error_2.append(hidden_layer_input* (1 - hidden_layer_input) * output_with_error)

                for j in range(self.number_of_nodes):
                    for i in range(self.num_output_nodes):
                        self.Wh2[i][j] = self.Wh2[i][j] + learning_rate * error_1[i] * self.hidden_layer[j]
                    for k in range(self.number_of_nodes):
                        self.Wh1[j][k] = self.Wh1[j][k]   + learning_rate * error_2[j] * training_vector[k]
            #print('finish',self.s) # Vatsal this for just testing 
                 
        return epochs
    def accuracy(self, input_data):
        correct = 0
        for data in input_data.values:
            hidden_layer, output_layer = self.forward_propagation(self.normalize_array(list(data)[1:11], self.max_feature_value))
            if (np.argmax(output_layer) == list(data)[-1]):
                correct += 1
        self.s=correct / len(input_data)
        return correct / len(input_data)
    def confusion_matrix(self,input_data):
        number_of_correct=0
        accuracy=0
        for value in input_data.values:
            training_vector = self.normalize_array(list(value)[1:11], self.max_feature_value)
            hidden_layer,output_layer=self.forward_propagation(training_vector)
            model_output_pred=np.argmax(output_layer)
            actual_output=list(value)[-1]
            if(model_output_pred == actual_output):
                number_of_correct+=1
                self.True_Positive[actual_output]+=1
                j=0
                while j<8:
                    if j!=actual_output:
                        self.True_Negative[j]+=1
                    else:
                        self.False_Positive[model_output_pred]+=1
                        self.False_Positive[actual_output]+=1
                        for i in range(8):
                            if i!=model_output_pred or i!=actual_output:
                                self.False_Negative[i]+=1
                    j+=1
        accuracy = number_of_correct / len(input_data)
        return accuracy

    def confusion_matrix_create(self):
#         print("True Positive: "+str(self.True_Positive))
#         print("False Positive: "+str(self.False_Positive))
#         print("True Negative: "+str(self.True_Negative))      
#         print("False Negative: "+str(self.False_Negative))
        self.numpyArray = np.array([self.True_Positive,self.False_Positive,self.True_Negative,self.False_Negative]) 

        # generating the Pandas dataframe 
        # from the Numpy array and specifying 
        # name of index and columns 
        self.confusion_matrix_final = pd.DataFrame( data = self.numpyArray,  
                                                    index = ["True_Positive","False_Positive","True_Negative","False_Negative"],  
                                                    columns = ["Class 0","Class 1", "Class 2","Class 3","Class 4", "Class 5","Class 6", "Class 7"])




d=MLP()
d.split_data()
Weights_input_layer=d.initialize_weight(10,10)
Weights_hidden_layer=d.initialize_weight(8,10)
d.merge_data()
d.set_weights(Weights_input_layer,Weights_hidden_layer)
print("Number of epochs:",d.back_propagation(0.3, 0.85))
print("Final Weights ")
print("Hidden Layer Weight = ",d.Wh1)
print("Output Layer Weight = ",d.Wh2)

"""

Initial Weight :
Hidden Layer Weight =  [[ 5.59433318e-01  1.47063978e-02 -2.30852603e-01 -1.76286939e-01
  -1.49895760e-01  1.46541214e-01 -1.39187302e-01 -2.59482586e-01
   4.25600745e-01 -2.82612253e-01]
 [-1.36309622e-01  3.03019775e-01  2.98731513e-01 -2.23445067e-01
  -1.33695754e-01 -4.76434622e-01  1.38279869e-02 -3.38535685e-01
  -8.24503438e-02 -2.94847969e-01]
 [-7.94496343e-02 -3.07266641e-01 -8.85902060e-02  3.27209886e-01
   6.53717863e-02 -1.92627312e-01  3.34970134e-01  1.52696141e-01
  -5.30042449e-01 -1.10474208e+00]
 [ 6.29696944e-02 -9.36265366e-02  4.24923068e-01 -4.80177710e-01
   7.14514621e-02 -5.67175656e-03 -6.50769568e-02 -1.08220414e-02
  -9.57538049e-02  8.14722380e-04]
 [-8.03001761e-02  3.69029398e-01 -1.88361144e-01  1.85806150e-01
   1.93037091e-02 -6.64509958e-01 -2.30460953e-01  4.16203557e-01
   3.23824738e-01  1.65614950e-02]
 [ 4.91401166e-01 -3.62213383e-01 -4.30082182e-01  4.68692705e-01
  -6.89652518e-01  3.29139329e-01  2.08687892e-02  4.17278519e-01
   4.35996630e-02 -2.30417334e-01]
 [-8.41157573e-02  2.23877922e-01  7.68813216e-02 -1.94624717e-01
  -1.14313488e-01  1.08665615e-02 -1.87092571e-01 -2.01720833e-01
  -3.46596643e-01 -2.16668726e-01]
 [ 1.60590721e-01 -3.98890540e-01 -4.33915740e-02  4.29933711e-02
  -2.13433183e-01  6.55794061e-01 -4.31889659e-01  6.86414760e-02
   9.45401692e-02  2.70670086e-01]
 [-9.55015370e-01  3.85897620e-01 -3.10889771e-01 -1.50663007e-01
  -2.88419377e-01  9.85926103e-05 -2.08706218e-02  1.29339257e-01
   7.19581660e-02  2.89374016e-01]
 [-1.49640174e-01  5.34894795e-02  4.89496472e-01 -2.69580351e-01
   3.00285695e-01  3.22750150e-01  5.77210924e-01 -3.99239343e-01
  -9.57159335e-02  1.63679303e-02]]
Output Layer Weight =  [[-0.77704849  0.16590801  0.09797724  0.27377367  0.03657968  0.42620153
   0.39113326 -0.27482454 -0.4021463  -0.13713716]
 [ 0.10485274 -0.10394604 -0.05015433  0.24951887  0.22138541 -0.2378936
   0.53645804  0.09444678 -0.01377127  0.21504902]
 [ 0.10217726  0.16196833  0.05421083 -0.35642205  0.04797011  0.16972125
   0.48909951 -0.14610391  0.3528292   0.65324062]
 [-0.05115329  0.33938062  0.30728433 -0.09229036 -0.21927841  0.31546194
   0.06311742 -0.04783692 -0.25173589  0.28017412]
 [-0.38512129  0.50998718  0.36718539  0.10988834  0.16129291  0.01578178
  -0.25281255  0.05166626 -0.27921825  0.01273205]
 [-0.2586746   0.2810402  -0.47493044 -0.34610322  0.15067708  0.07041698
   0.41268473 -0.16329667 -0.09031942 -0.01460909]
 [ 0.31392784  0.0243479  -0.32770757  0.22026709  0.23306275 -0.2910481
  -0.14098494 -0.26040701  0.56467193  0.45277547]
 [ 0.12253963  0.1522326   0.3661648   0.02489514 -0.16148586  0.39053136
   0.16135947 -0.14183235 -0.04158232 -0.48576695]]
Number of epochs: 1342
Final Weights 
Hidden Layer Weight =  [[-0.01254347  0.27769931  1.01038264  0.64396206  0.44341531 -0.3005272
   0.54773947 -0.24111447 -0.24108449 -0.25500427]
 [-3.89605929  1.48171119 -1.19444056 -0.97447689 -0.49014644 -1.32382528
   1.86253133  1.93623577  0.23347265 -2.71596317]
 [-1.53301941  0.68805779 -0.67327923  2.11773687 -0.30757901 -0.12351454
   3.84156574 -0.79049152 -4.48837974 -2.7468723 ]
 [-3.51853395 -0.17429873  0.8066119  -5.66629989 -0.15135577  0.52182001
  -1.4358877   1.06888509 -0.53034382  1.42110172]
 [-1.53743534  1.94391105 -0.61165326  4.83973949 -0.02871267 -1.53961499
   3.91284624 -0.38130997 -5.96400173  0.39501418]
 [ 1.32727592 -1.94661869 -5.84003627  0.03802762 -0.45237566  1.65364713
  -2.45477329 -0.5626947   1.50276051 -0.41688501]
 [ 0.48800922 -0.2593348  -0.8728992   0.47028897  2.66928136 -1.74756409
  -0.59720851 -5.6416641  -4.56314149  2.25789349]
 [-1.30687551 -0.03051417 -0.72978021 -0.61794013  1.65149602  0.94043426
  -3.22476701 -1.60411811 -2.87756461  4.00515611]
 [-1.93598497  2.69258434 -3.61501563 -2.06118795  0.25542633  0.9896243
  -1.96404619  2.83750714 -0.75598004  0.60843877]
 [ 0.43464492 -1.55031794 -1.01192283 -0.79531825  0.80960542  0.26641097
  -1.49004538 -0.85280005  2.9864934  -1.82145104]]
Output Layer Weight =  [[-1.02010673  0.04846913 -0.40148031  0.13474293  0.20136641  0.25076522
  -0.02781928 -0.36169814 -0.01065332 -0.27382219]
 [-1.64178322 -1.53141585  0.59774339  0.8448978   0.96690443 -1.52153451
  -0.93542809 -0.09166736 -0.40560551  1.90697952]
 [-0.46351388 -0.48368548  0.13104939 -3.30081743 -0.60121751 -0.99649408
   4.44521523 -2.59477387  2.77273033 -0.57693012]
 [ 1.90828827  0.8058356   1.76357955 -1.68016627 -2.7613698   3.75984199
   0.16207355  0.09556261 -1.05053399 -3.21780832]
 [-1.77853999 -1.14748427  1.80608159 -1.05309447  0.0362738  -2.03165478
  -2.71499378  3.11111276 -0.48309779  1.76243971]
 [-1.66624466  3.53817606 -4.26589613  2.02638179  2.44894653 -0.37534546
  -1.30789597  0.67689667 -1.9325488   0.11900879]
 [-0.91171469 -0.4431407   0.01170327 -0.31014539 -0.1882683  -1.41201094
   0.35520836 -0.6063648   0.59154695  0.6083707 ]
 [-1.02294619 -1.03854369  0.70108238  3.21241745  0.15133211  0.29859719
   0.12224827 -0.83700928 -0.26224616 -0.46376878]]
"""

print("Holdout Accuracy: " + str(d.confusion_matrix(d.test_data)*100))
#print("Training Accuracy: " + str(accuracy(Training_Set)))

d.confusion_matrix_create()
sns.heatmap(d.confusion_matrix_final,cmap = 'Blues', linewidths=0.5, annot=True)
print("Holdout Accuracy: " + str(d.confusion_matrix(d.df1)*100))
