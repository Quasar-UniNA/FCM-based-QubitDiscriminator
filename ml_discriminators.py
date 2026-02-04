import numpy as np
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from qiskit.quantum_info.analysis import hellinger_fidelity
from copy import deepcopy
import matplotlib.pyplot as plt

class sklearn_discriminator():
    def __init__(self,  processor_size, training_data:dict, skl_algorithm) -> None:
        
        self.training_data={'pq'+str(i):'' for i in range(processor_size)}
        self.disc = {'pq'+str(i):deepcopy(skl_algorithm) for i in range(processor_size)}
        self.figs = {'pq'+str(i):'' for i in range(processor_size)}
        
        for i in training_data.keys():
            #Check if the i-th physical qubit is in the calibration data
            IQ_0, IQ_1 =  training_data[i][0], training_data[i][1]
            real_0, img_0 = np.real(IQ_0), np.imag(IQ_0)
            y_0 =  np.zeros(real_0.shape[0],)
            data_0 =  np.column_stack((real_0, img_0, y_0))
            real_1, img_1 = np.real(IQ_1), np.imag(IQ_1)
            y_1 =  np.ones(real_1.shape[0],)
            data_1 =  np.column_stack((real_1, img_1, y_1))
            self.training_data['pq'+str(i)] = np.concatenate((data_0, data_1 ), axis=0)
            
    
    def train(self, plot=False):
        for pq, data in zip(self.training_data.keys(), self.training_data.values()):
            if isinstance(data, np.ndarray):
                X, Y =  data[:,:2], data[:,2]
                self.disc[pq].fit(X, Y)
        
                if plot:
                    self.figs[pq], ax = plt.subplots()
                    # title for the plots
                    title = ('Decision surface of linear SVC ')
                    # Set-up grid for plotting.
                    X0, X1 = X[:, 0], X[:, 1]
                    xx, yy = self.make_meshgrid(X0, X1)

                    self.plot_contours(ax, self.disc[pq], xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
                    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=10, edgecolors='k')
                    ax.set_ylabel('Q (arb. units)')
                    ax.set_xlabel('I (arb. units)')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title('Decison surface ' + pq)
                    ax.legend()


    def predict(self, n_qubits, IQ_data, layout:list):
        IQ_data_qubits = {}
        self.prediction = {}
        for i in range(n_qubits):
            IQ_data_qubits['q'+str(i)]=  np.column_stack((np.real(IQ_data[:,i]), np.imag(IQ_data[:,i])))
            self.prediction['q'+str(i)]= self.disc['pq'+str(layout[i])].predict(IQ_data_qubits['q'+str(i)])

    def predict_proba(self, n_qubits, IQ_data, layout:list):
        IQ_data_qubits = {}
        self.prediction = {}
        for i in range(n_qubits):
            IQ_data_qubits['q'+str(i)]=  np.column_stack((np.real(IQ_data[:,i]), np.imag(IQ_data[:,i])))
            self.prediction['q'+str(i)]= self.disc['pq'+str(layout[i])].predict_proba(IQ_data_qubits['q'+str(i)])
        return self.prediction

    def get_counts(self, probs=False):
        def reverse_string(s):
            return s[::-1]
        output_strings = np.array([''.join(str(int(arr[i])) for arr in self.prediction.values()) for i in range(self.prediction['q0'].shape[0])])
        reversed_bool_strings = np.vectorize(reverse_string)(output_strings)
        counts = Counter(reversed_bool_strings)
        if probs:
            basis = [_ for _ in counts.keys() if _ != 'None']
            norm = sum([counts[i] for i in basis])
            for state in basis:
                counts[state] = counts[state]/norm 
        return counts
    
    
    def make_meshgrid(self, x, y, h=500000):
        x_min, x_max = x.min() - 1000000, x.max() + 1000000
        y_min, y_max = y.min() - 1000000, y.max() + 1000000
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(self, ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def plot(self, qubits:list):
        for _ in qubits:
            handles = []
            labels = ['0', '1']
            colors = [plt.cm.coolwarm(0), plt.cm.coolwarm(0.99)]  # Colors for the classes
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)   for label, color in zip(labels, colors)]  
            self.figs['pq'+str(_)].legend(handles=handles)
            self.figs['pq'+str(_)].show()





