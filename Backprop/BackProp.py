import numpy as np
import numpy.random as rnd
import json, sys

jsonCfgPath = "CfgEx.json"

class Layer:
    # dim  -- number of neurons in the layer
    # prev -- prior layer, or None if input layer.  If None, all following
    #  params are ignored.
    # act -- activation function: accept np.array of z's; return
    #  np.array of a's
    # act_prime -- derivative function: accept np.arrays of z's and of a's,
    #  return derivative of activation wrt z's, 1-D or 2-D as appropriate
    # wgts -- initial weights. Set to small random if "None". (between -.5 and .5)
    #
    # All the following member data are None for input layer
    # in_wgts -- matrix of input weights
    # in_derivs -- derivatives of E/weight for last sample
    # zs -- z values for last sample
    # z_derivs -- derivatives of E/Z for last sample
    # batch_derivs -- cumulative sum of in_derivs across current batch
    def __init__(self, dim, prev, act, wgts = None):        # act_prime???
        self.dim = dim
        self.prev = prev
        self.act = act
        # self.act_prime = act_prime
        self.wgts = wgts
        #
        # self.in_wgts = None if prev is None else prev.wgts
        self.in_derivs = None
        self.z_derivs = None
        self.z_derivs = None
        self.batch_derivs = None
        self.outputs = None
        self.nxt = None

    def activate(self):
        if (self.act == "relu"):
            self.outputs = actRelu(self.outputs)
        elif (self.act == "softmax"):
            self.outputs = actSoftmax(self.outputs)


    def get_dim(self):
        return self.dim
    
    # Return the dE/dW for the weight from previous layer's |src| output to
    # our |trg| output. 

    # def get_deriv(self, src, trg): ######################################################################################


   
    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals = None):
        inputs = vals if vals is not None else self.prev.outputs

        while (self.wgts is not None and len(inputs) < len(self.wgts[0])):
            inputs = list(inputs)
            inputs.append(1)

        self.outputs = np.dot(self.wgts, inputs) if self.wgts is not None else inputs
        self.activate()

    
    # Compute self.in_derivs, assuming 
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays 
    #       of outputs and of labels, and returns an np array 
    #       of dE/da for each output
    def backpropagate(self, err_prime=None, labels=None):
        if (self.act == "softmax"):
            backSoftmax(self, labels)
        elif (self.act == "relu"):
            backRelu(self.nxt)


    # Adjust all weights by avg gradient accumulated for 
    #  current batch * -|rate|

    # def apply_batch(self, batch_size, rate): ######################################################################################
     
    # Reset internal data for start of a new batch

    # def start_batch(self): ######################################################################################

    # Add delta to the weight from src node in prior layer
    # to trg node in this layer.

    # def tweak_weight(self, src, trg, delta): ######################################################################################

    def validate(self, net, data, lyrNum):
        if (self.wgts is not None):
            for c in range(len(self.wgts)):
                for d in range(len(self.wgts[c])):
                    self.wgts[c][d] = self.wgts[c][d] * 1.01
                    old = net.error
                    net.run_batch(data, 1)
                    new = net.error
                    log("test " + str(lyrNum - 1)) # fill in rest of line
                    self.wgts[c][d] = self.wgts[c][d] / 1.01

    # Return string description of self for debugging

    # def __repr__(self): ######################################################################################
      

class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
            # skip any terms in the crossentropy or entropy summations for which the label is zero.
    # wgts -- list of one 2-d np.array per layer in arch
            # Note that each set of weights has a different dimension, appropriate to its layer. This means you 
            # can't convert the whole thing into one big numpy array, though each element may be so converted.
    def __init__(self, arch, err, wgts = None):
        self.err = err
        self.output = None
        self.error = None

        self.lyrs = [Layer(arch[0][0], None, None)]

        for i in range (len(arch) - 1):
            self.lyrs.append(Layer(arch[i + 1][0], self.lyrs[i], arch[i + 1][1], wgts[i]))
            self.lyrs[i].nxt = self.lyrs[i + 1] # self.lyrs[i].nxt
    
    # Forward propagate, passing inputs to first layer, and returning outputs
    # of final layer
    def predict(self, inputs):
        self.lyrs[0].propogate(inputs) # self.lyrs[0].propogate(inputs)

        for i in range(len(self.lyrs) - 1):
            self.lyrs[i + 1].propogate()

            return self.lyrs[-1].outputs

        
    # Assuming forward propagation is done, return current error, assuming
    # expected final layer output is |labels|
    def get_err(self, labels):
        y = labels
        a = self.output
        n = a.shape[0]
        return -np.sum(y * np.log(a) + np.subtract(1, y) * np.log(1 - a)) / n


    
    # Assuming a predict was just done, update all in_derivs, and add to
    # batch_derivs
    def backpropagate(self, labels):
        for i in range(len(self.lyrs)):
            self.lyrs[-i].backpropogate(labels = labels)

    
    # Verify all partial derivatives for weights by adding an
    # epsilon value to each weight and rerunning prediction to
    # see if change in error correctly reflects weight change
    def validate_derivs(self, inputs, outputs):
        for i in range(len(self.lyrs)):
            self.lyrs[i].validate(self, data, i)

    
    # Run a batch, assuming |data| holds input/output pairs comprising the 
    # batch. Forward propagate for each input, record error, and backpropagate.
    # At batch end, report average error for the batch, and do a derivative 
    # update.
    def run_batch(self, data, rate):
        self.output = self.predict(data[0][0])
        self.error = self.get_err(data[0][1])
        self.backpropogate(data[0][1])

            
def main(cmd, cfg_file, data_file):
    # newLog()

    cfg = open(cfg_file, "r") # Obtain file contents
    cfgParams = json.load(cfg)

    data = open(data_file, "r") # Obtain file contents
    dataParams = json.load(data)


    net = Network(cfgParams['arch'], cfgParams['err'], wgts = cfgParams['wgts'])

    # Close files
    # cfg.close()
    # data.close()

    if (cmd == "verify"):
        verify(net, dataParams)
    elif (cmd == "run"):
        net.run_batch(dataParams, 1)
    else:
        print("Improper command given")


    # endLog()

def actRelu(vals):
    return np.maximum(0, vals)

def actSoftmax(vals):
    sum_vals = np.exp(vals - np.max(vals))
    return sum_vals / sum_vals.sum()

def backRelu(lyr):
    d = np.dot(np.transpose(lyr.wgts), lyr.in_derivs)
    lyr.prev.in_derivs = d

def backSoftmax(lyr, labels):
    y = labels
    a = lyr.outputs

    dEda = []
    daidaj = []
    for c in range(len(a)):
        dEda.append(-y[c]/a[c] if a[c] != 0 else 0)
        daidaj.append([])
        for d in range(len(a)):
            daidaj[c].append(dada(a[c], a[d]))

    lyr.in_derivs = np.dot(daidaj, dEda)
    lyr.batch_derivs.append(lyr.in_derivs)

def dada(ai, aj):
    if(ai == aj):
        return ai - np.square(ai)
    else:
        return -ai * aj

# def endLog():
    # unfilled

# def log(msg):
    # unfilled

# def newLog():
    # unfilled

# def openJSON(filePath):
#    with open(filePath) as file:
        # unfilled

# accepts three commandline arguments: a command "run" or "verify", and two files,
# the first a network configuration file and the second a file of input/output pairs.
main(sys.argv[1], sys.argv[2], sys.argv[3])



# 12 loops total, including all I/O
# 2 uses of np.outer
# 3 uses of np.dot              x
# 1 use of np.diag
# 1 use of np.transpose         x
# 3 uses of np.sum
# 2 uses of np.log              x
# 1 use of np.exp               x

# 227 lines total, including about 75 comment/blank lines