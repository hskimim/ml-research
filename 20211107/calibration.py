import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

class CalibrationLoss : 
    
    def __init__(self, strategy='uniform', n_bins=10, func='mean', dist='l1') : 
        func_map = {
        'mean' : np.mean,
        'max' : np.max,
        'min': np.mean,
        'median':np.median
        }
        aggregator = func_map.get(func)
        if aggregator is None : 
            raise ValueError("Invalid func : please choose func among {}".format(func_map.keys()))

        dist_map = {
            'l1' : lambda x,y : np.abs(x-y),
            'l2' : lambda x,y : np.sqrt((x-y)**2),
        }
        measure = dist_map.get(dist)
        if measure is None : 
            raise ValueError("Invalid func : please choose func among {}".format(dist_map.keys()))
        
        self._strategy = strategy
        self._n_bins = n_bins
        self._aggregator = aggregator
        self._measure = measure
        
    def calculate_ce(self, pred, true, visualize=False, return_classwise=False) : 
        unique_l = range(pred.shape[1])
        loss_container = np.full(shape=(len(unique_l),), fill_value=np.nan)
        
        for idx in range(len(unique_l)) : 

            # One Vs Rest Scheme
            tmp_true = true * np.nan
            tmp_true[true == unique_l[idx]] = 1
            tmp_true[true != unique_l[idx]] = 0

            prob_true, prob_pred = calibration_curve(tmp_true, pred[:,unique_l[idx]],strategy=self._strategy, n_bins=self._n_bins)
            if visualize : 
                plt.scatter(prob_pred, prob_true)
                plt.show()
                
            l = self._aggregator(self._measure(prob_true, prob_pred))
            loss_container[idx] = l

        return loss_container if return_classwise else np.mean(loss_container)
    
    def caculate_log_loss(self, pred, true) : 
        return log_loss(true, pred)