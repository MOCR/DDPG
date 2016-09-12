from StateEstimator import StateEstimator
from StateEstimatorHyb import StateEstimatorHyb
from StateEstimatorNoFeedBack import StateEstimatorNoFeedBack
from StateEstimatorRegression import StateEstimatorRegression

StateEstimatorType={'Inv':StateEstimator, 
                    'Hyb':StateEstimatorHyb, 
                    'NoFeedBack':StateEstimatorNoFeedBack,
                    'Regression':StateEstimatorRegression}