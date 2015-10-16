from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf
#from linear_regression import LinearRegressionModel
import numpy as np



# set the location of features
train_location = 'file:///Users/sradhakr/Desktop/Assignment4/Assignment4/data/train_word2vec_100'
test_location = 'test_word2vec_100'

# start spark context
conf = (SparkConf().set("log4j.rootCategory", "Warn, concole"))
sc = SparkContext(conf=conf)

# load train/val/test features
data_training = sc.pickleFile(train_location)
data_test = sc.pickleFile(test_location)

# convert our train and test data to LabeledPoints
# I also repartition as I was running the code on a machine with 8 cores.
labeled_training = data_training.map(lambda x: LabeledPoint(x[1], x[0])).repartition(8).cache()
labeled_test = data_test.map(lambda x: LabeledPoint(x[1], x[0])).repartition(8).cache()

num_ftr = 100
reg_param = 0.001

lrm = LinearRegressionModel()

################
### Code me! ###
################

cost, grad = lrm.cost_grad_rdd(labeled_training)
print cost
print grad


# Question 2: Double check the gradient.

index = 50
epsilon = 0.001

lrm2 = LinearRegressionModel()
lrm2.update_w(index,epsilon)
cost_addl,grad_addl=lrm2.cost_grad_rdd(labeled_training)

grad_calculated = (cost_addl-cost)/epsilon
print "Cost difference"
print grad_calculated

# Question 3: Find the optimum value for the objective function of ridge regression.
N = labeled_training.count()
I = np.identity(num_ftr)
X = np.matrix(labeled_training.map(lambda lp: lp.features).collect())
t = np.matrix(labeled_training.map(lambda lp: lp.label).collect())
#reg_param=0.0
w_optimal = np.linalg.inv(reg_param*N*I + (X.T*X))*X.T*t.T

lrm3 = LinearRegressionModel(reg_param)
lrm3.set_w(np.squeeze(np.asarray(w_optimal)))
cost_optimal,grad_optimal=lrm3.cost_grad_rdd(labeled_training)
# cost = 0.75899 at reg 0.001

# Question 4: Use the gradient descent algorithm to train the model.

lrm4 = LinearRegressionModel(reg_param)
steps = [pow(10,-4),pow(10,-3), pow(10,-2), pow(10,-1), pow(10,0), pow(10,1), pow(10,2), pow(10,3), pow(10,4) ]

costs = []
for step in steps:
	costs_list = lrm4.train_gd(labeled_training, step)
	costs.append(costs_list[-1])
	
print "step"
print steps
print "cost"
print costs

# Question 5: Use the conjugate gradient decent algorithm.

# Question 6: Use the L-BFGS algorithm.

# Question 7: Use the batch L-BFGS algorithm.

# Question 8: Visualize the values of the objective function for all algorithm at each iteration.
# Use semilogy to show the differences better.

sc.stop()
