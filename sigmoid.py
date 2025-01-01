from typing import List
from numpy.typing import NDArray
import numpy as np
import math

def sigmoid_func(prediction:NDArray[np.float64])->List:
    """
    Applies the sigmoid function to a list of predictions and returns a binary classification result (0 or 1).

    The sigmoid function is applied element-wise to each value in the input array `prediction`. 
    If the sigmoid output is greater than 0.5, the corresponding prediction is classified as 1, 
    otherwise, it is classified as 0.

    Args:
        prediction (NDArray[np.float64]): A numpy array or list of numerical prediction values

    Returns:
        List: A list of binary classification results (0 or 1) corresponding to each element in `prediction`.
    """
    
    pred_sigmoid:List[int]=[]
    
    for pred in prediction:
        
        sigmoid:float=1/(1+math.exp(-1*pred))
        if sigmoid>=0.5:
            pred_answer:int=1   
        else:
            pred_answer:int=0
            
        pred_sigmoid.append(pred_answer)
            
    return pred_sigmoid

if __name__=="__main__":
    
    test=sigmoid_func([0.81,0.0012,0.28])
    print(test)