import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture


# --- Unsupervised GMM Discriminator for Single Qubit scenario ---
class GMM_discriminator_unsupervised:
    def __init__(self, processor_size, training_data):
        self.fitted_probs = []
        self.fitted_mapping = []
        self.n_shots = 0
        self.current_layout = None

    def train(self):
        pass # Unsupervised: ignore training data

    def predict(self, n_qubits, IQ_data, layout):
        self.current_layout = layout
        self.n_shots = IQ_data.shape[0]
        self.fitted_probs = []
        self.fitted_mapping = []

        for k, q_idx in enumerate(layout):
            q_data = IQ_data[:, k]
            X_test = np.column_stack((q_data.real, q_data.imag))
            
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X_test)
            
            self.fitted_probs.append(gmm.predict_proba(X_test))
            
            # Heuristic: smaller real mean = state '0'
            if gmm.means_[0][0] < gmm.means_[1][0]:
                self.fitted_mapping.append({0: '0', 1: '1'})
            else:
                self.fitted_mapping.append({0: '1', 1: '0'})

    def get_counts(self, threshold=0.0):
        counts = Counter()
        discarded_count = 0
        for i in range(self.n_shots):
            shot_result = []
            discard_shot = False
            for k in range(len(self.current_layout)):
                probs = self.fitted_probs[k][i]
                if np.max(probs) < threshold:
                    discard_shot = True
                    break
                shot_result.append(self.fitted_mapping[k][np.argmax(probs)])
            
            if discard_shot:
                discarded_count += 1
            else:
                # Qiskit ordering: reversed
                counts["".join(reversed(shot_result))] += 1
        counts['None'] = discarded_count
        return dict(counts)