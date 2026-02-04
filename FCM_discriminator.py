import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
from qiskit.quantum_info.analysis import hellinger_fidelity


class fcm_discriminator_v2():
    def __init__(self,  processor_size, training_data:dict) -> None:

        '''training_data = {physical_qubit_index: (IQ_data_0, IQ_data_1)}'''
        
        self.training_data={'pq'+str(i):'No Data' for i in range(processor_size)}
        for i in training_data.keys():
            IQ_0, IQ_1 =  training_data[i][0], training_data[i][1]
            #IQ_0, IQ_1 = data_0s[:,layout_0s.index(i)],data_1s[:,layout_1s.index(i)]
            real_0, img_0 = np.real(IQ_0), np.imag(IQ_0)
            real_1, img_1 = np.real(IQ_1), np.imag(IQ_1)
            self.training_data['pq'+str(i)] = np.column_stack((real_0, img_0, real_1, img_1))

        self.processor_size = processor_size
        self.physical_qubits_info = {'pq'+str(i): '' for i in range(processor_size)}

    def compute_readout_error(self, qubit_index, cluster_membership, labels):
        shots = self.training_data['pq'+str(qubit_index)].shape[0]
        for state in labels.keys():
            if state==0:
                correct_prediction_0 = np.sum(cluster_membership[:shots] == labels[state])
                error_0 =  1 - (correct_prediction_0/shots)
            if state==1:
                correct_prediction_1 = np.sum(cluster_membership[shots:] == labels[state])
                error_1 =  1 - (correct_prediction_1/shots)
        
        return error_0, error_1


    def train(self, error=0.00005, maxiter=5000, init=None, **kwargs):
        '''Build clusters on training data.

        kwargs: 
            plot:list -> Indices of physical qubits to plot. 
        
        '''
        self.iq_plots = {'pq'+str(i):' ' for i in range(self.processor_size)}
        if 'plot' in kwargs: qubits_to_plot = kwargs['plot']

        for qubit_index in range(self.processor_size):
            #Check that calibration data for this physical qubit are present
            if isinstance(self.training_data['pq'+str(qubit_index)], np.ndarray):

                # Prepare data in the correct format 
                data_0, data_1 =  np.vstack((self.training_data['pq'+str(qubit_index)][:,0], self.training_data['pq'+str(qubit_index)][:,1])), np.vstack((self.training_data['pq'+str(qubit_index)][:,2], self.training_data['pq'+str(qubit_index)][:,3]))
                data = np.concatenate((data_0, data_1 ), axis=1)

                #Perform Clustering  
                cntr, u_train, u0_train, d_train, jm_train, p_train, fpc_train = fuzz.cluster.cmeans(data, 2, m=2, error=error, maxiter=maxiter, init=init)    
                cluster_info = {'cntr':cntr, 'u_train':u_train, 'u0_train':u0_train, 'd_train':d_train, 
                                                        'jm_train':jm_train, 'p_train':p_train, 'fpc_train':fpc_train}
                
                # Define labels to understand rows in the cluster matrix
                mean_data = {0:np.mean(data_0, axis=1),1:np.mean(data_1, axis=1)}
                labels = {0:'', 1:''}
                for state in mean_data.keys():
                    distances = []
                    for r in range(2): 
                        distances.append(np.linalg.norm(cluster_info['cntr'][r]-mean_data[state]))
                    labels[state] = np.argmin(distances)
                
                self.physical_qubits_info['pq'+str(qubit_index)] = {'cluster_info': cluster_info, 'labels':labels }
                cluster_membership = np.argmax(cluster_info['u_train'], axis=0)
                self.physical_qubits_info['pq'+str(qubit_index)]['readout_errors'] = self.compute_readout_error(qubit_index=qubit_index, cluster_membership=cluster_membership, labels=labels)

                if 'plot' in kwargs and qubit_index in qubits_to_plot:
                    pqi = 'pq' + str(qubit_index)
                    cal_data = self.training_data[pqi]
                        
                    if isinstance(cal_data, np.ndarray):
                        fig, axs = plt.subplots(1,3, figsize=(15, 8))
                        axs[0].scatter(cal_data[:,0], cal_data[:,1], s=1, marker='.', c='red', label='0')
                        axs[0].scatter(cal_data[:,2], cal_data[:,3], s=1, marker='.', c='blue', label='1')
                        axs[0].set_xlabel("I")
                        axs[0].set_ylabel("Q")
                        axs[0].legend()
                        axs[0].set_title("Calibration Data " + pqi)

                        colors = {0:'red', 1:'blue'}
                        for j in range(2):
                            axs[1].scatter(data[0,:][cluster_membership == labels[j]],
                                    data[1,:][cluster_membership == labels[j]], marker='.', color=colors[j], s=1, label=str(j))
                        
                        # Mark the center of each fuzzy cluster
                        for pt in cntr:
                            axs[1].plot(pt[0], pt[1], 'rs')

                        axs[1].set_xlabel("I")
                        axs[1].set_ylabel("Q")
                        axs[1].legend()
                        axs[1].set_title("Cluster IQ Qubit " + pqi)

                        

                        self.iq_plots[pqi] = (fig, axs)
        
            else: self.physical_qubits_info['pq'+str(qubit_index)] = 'No Data'
        


    def predict(self, n_qubits, IQ_data, layout:list, error=0.00005, maxiter=5000, init=None, **kwargs):
        self.layout = layout
        if 'plot' in kwargs: qubits_to_plot = kwargs['plot']
        IQ_data_qubits = {}
        for i in range(n_qubits):
            IQ_data_qubits['q'+str(i)]=  np.vstack((np.real(IQ_data[:,i]), np.imag(IQ_data[:,i])))
        self.prediction={}
        for qubit_index in range(n_qubits):
            qubit = 'q'+str(qubit_index)
            pqi = 'pq'+str(layout[qubit_index])
            self.prediction[qubit] = fuzz.cmeans_predict(IQ_data_qubits[qubit], cntr_trained=self.physical_qubits_info[pqi]['cluster_info']['cntr'],  m=2, error=0.00005, maxiter=5000, init=None)[0]
            if 'plot' in kwargs and layout[qubit_index] in qubits_to_plot:
                fig_pred = self.iq_plots[pqi][0]
                axs = self.iq_plots[pqi][1]
                axs[2].scatter(IQ_data_qubits[qubit][0,:], IQ_data_qubits[qubit][1,:], marker='.', c='green', label='0')
                axs[2].set_title("Logical Qubit " + str(qubit_index))
                fig_pred.tight_layout()


    def get_counts(self, probs=False, **kwargs):
        diff_mem, q_state = {}, {}
        if 'automatic_threshold' in kwargs:
            automatic_threshold = kwargs['automatic_threshold']
        else: automatic_threshold=False

        for qubit_index in range(len(self.prediction.keys())):
            qubit = 'q'+str(qubit_index)

            if 'd' in kwargs and not automatic_threshold: 
                d = kwargs['d']
                if isinstance(d, dict): threshold = d[qubit]
                else: threshold = d

            if 'd' not in kwargs and not automatic_threshold: threshold=0

            if automatic_threshold:
                threshold =  sum(self.physical_qubits_info['pq'+str(self.layout[qubit_index])]['readout_errors']) 
            
            print('Threshold for logical qubit '+ str(qubit_index)+ ' mapped in pq '+ str(self.layout[qubit_index]) +' = '+ str(threshold))

            diff_mem[qubit] = self.prediction[qubit][0,:] - self.prediction[qubit][1,:] 
            diff_mem[qubit] = np.where(np.abs(diff_mem[qubit]) > threshold, diff_mem[qubit], np.nan)
            mask_negative_q = np.isnan(diff_mem[qubit]) | (diff_mem[qubit] < 0)
            mask_nan_q = np.isnan(diff_mem[qubit])
            q_state[qubit_index] = np.where(mask_negative_q, 1, 0).astype(float)
            q_state[qubit_index][mask_nan_q] = np.nan
        
        counts = {'None':0}
        for bits in zip(*q_state.values()):
            if any(np.isnan(x) for x in bits):
                counts['None'] = counts['None'] + 1

            else:
                string = ''
                for qubit_index in range(len(self.prediction.keys())):
                    qubit = 'q'+str(qubit_index)
                    string = str(self.physical_qubits_info['pq'+str(self.layout[qubit_index])]['labels'][bits[qubit_index]])+ string
                
                if string in counts.keys():
                    counts[string] = counts[string] + 1
                else:
                    counts[string] = 1

        if probs:
            basis = [_ for _ in counts.keys() if _ != 'None']
            norm = sum([counts[i] for i in basis])
            for state in basis:
                counts[state] = counts[state]/norm 

        
        return counts
    
        



