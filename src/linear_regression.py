import numpy as np


class LinearRegressionModel():
    def __init__(self, reg_param=reg_param, num_ftr=100):
        self.regParam = reg_param
        self.w = np.zeros((num_ftr,))
    def update_w(self, index, epsilon):
        self.w[index] += self.w[index]+epsilon   
    def set_w(self, w_optimal):
        self.w = w_optimal
    def update_param(self, update, step_size):
        self.w += update * step_size
    def cost_grad_sample(self, point):
        """
        Computes cost and gradient for a data point. Note that
         point.features and point.label contains the feature vector
         and target value respectively.
        """
        ################
        ### Code me! ###
        ################     
        cost = (np.dot(self.w, point.features) - point.label)**2
        grad = (np.dot(self.w, point.features) - point.label)*point.features
        # return cost, grad and a constant one that will be used for counting
        return (cost, grad, 1)
    def cost_grad_rdd(self, labeled_points):
        """
        Computes cost and gradient on the whole labeled_points RDD.
         You will need to use the cost_grad_sample() method.
        """
        ################
        ### Code me! ###
        ################
        cost_grad_count_rdd = labeled_points.map(lambda p: self.cost_grad_sample(p)).cache()
        cost, grad, count = cost_grad_count_rdd.reduce(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2]))        
        cost = cost/(2*count) +(self.regParam*(np.dot(self.w, self.w)))/2        
        grad = self.regParam*self.w + grad/count     
        return (cost, grad)
    def predict_target(self, labeled_points):
        """
        Predicts target value for all the data points in labeled_points
        """
        return labeled_points.map(lambda p: (np.dot(self.w, p.features), p.label))
    def train_gd(self, labeled_points, step, iterations=100):
        """
        Performs gradient descent to train the parameters of the regression model.
        Returns costs a list of objective function values at each iteration.
        """
        ################
        ### Code me! ###
        ################
        k = 0
        w=self.w
        costs = []
        while k < iterations:
        	cost,grad=self.cost_grad_rdd(labeled_points)
        	costs.append(cost)
        	self.update_param(-grad,step)
        	k = k+1
        return costs
    def train_cg(self, labeled_points, step, iterations=100):
        """
        Performs conjugate gradient descent to train the parameters of the regression model.
        Returns costs a list of objective function values at each iteration.
        """
        ################
        ### Code me! ###
        ################
        return costs
    def train_lbfgs(self, labeled_points, step, iterations=100):
        """
        Performs L-BFGS algorithm to train the parameters of the regression model.
        Returns costs a list of objective function values at each iteration.
        """
        ################
        ### Code me! ###
        ################
        return costs
    def train_batch_lbfgs(self, labeledPoints, step, iterations=100, miniBatchFraction=0.1):
        """
        Performs conjugate gradient descent to train the parameters of the regression model.
        Returns costs a list of objective function values at each iteration.
        """
        ################
        ### Code me! ###
        ################
        return costs

