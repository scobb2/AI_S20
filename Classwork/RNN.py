import numpy as np

timesteps = 100                                                      
input_features = 10                                                
output_features = 20                                 

# Features per time step
inputs = np.random.random((timesteps, input_features))         

state_t = np.zeros((output_features,))                   

# start with random values as mock "trained net"
W = np.random.random((output_features, input_features))            
U = np.random.random((output_features, output_features))              
b = np.random.random((output_features,))                             

successive_outputs = []
for input_t in inputs:                                                
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)   

    successive_outputs.append(output_t)                                

    state_t = output_t                                           

final_output_sequence = np.concatenate(successive_outputs, axis=0)