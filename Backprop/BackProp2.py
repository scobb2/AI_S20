import numpy as np
import numpy.random as rnd
import json, sys

dEdzs = []
ws = []

class Layer:
    # dim  -- number of neurons in the layer
    # prev -- prior layer, or None if input layer.  If None, all following
    #  params are ignored.
    # act -- activation function: accept np.array of z's; return
    #  np.array of a's
    # act_prime -- derivative function: accept np.arrays of z's and of a's,
    #  return derivative of activation with regard to z's, 1-D or 2-D as appropriate
    # weights -- initial weights. Set to small random if "None".
    #
    # All the following member data are None for input layer
    # in_weights -- matrix of input weights
    # in_derivs -- derivatives of E/weight for last sample
    # zs -- z values for last sample
    # z_derivs -- derivatives of E/Z for last sample
    # batch_derivs -- cumulative sum of in_derivs across current batch
    def __init__(self, dim, prev, act, weights = None):
        self.dim = dim
        self.prev = prev
        if prev is None:
            return
        
        self.act = act

        if (self.act == "relu"):
            self.act_prime = "backRelu"
        elif (self.act == "softmax"):
            self.act_prime = "backSoftmax"

        self.in_weights = weights if weights is not None else (rnd.random((dim, prev.dim + 1)).astype(np.float32) - 0.5)
        self.in_derivs = np.zeros((dim, prev.dim + 1), dtype = np.float32)
        self.zs = np.zeros((dim), dtype = np.float32)
        self.z_derivs = np.zeros((dim), dtype = np.float32)
        # self.batch_derivs = np.zeros((dim, prev.dim + 1), dtype = np.float32)
        self.batch_derivs = 0


    def get_dim(self):
        return self.dim
    
    # Return the dE/dW for the weight from previous layer's |src| output to
    # our |trg| output.
    def get_deriv(self, src, trg):
        dEdW = np.dot(src.outputs, trg.z_derivs)
        return dEdW

    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals = None):
        if vals is not None:
            self.outputs = vals
        else:
            inputs = list(self.prev.outputs)
            inputs.append(1) # adding a 1 to the end of inputs to mult against the bias weights in the weight list
            print("len of input list: " + str(len(inputs)) + "    len of wgt list: " + str(len(self.in_weights)))

            print("inputs: " + str(inputs) + "   weights: " + str(self.in_weights))
            self.zs = np.dot(inputs, np.transpose(self.in_weights)) ## dot(self.prev.outputs, self.prev.in_weights)
            print("outputs (zs): " + str(self.zs))
            if (self.act == "relu"):
                self.outputs = self.relu(self.zs)
                print("outputs after act (as): " + str(self.outputs))
            else:
                self.outputs = self.softmax(self.zs)
                print("outputs after act (as): " + str(self.outputs))


            # self.prev.outputs = list(self.prev.outputs)
            # self.prev.outputs.append(1) # adding a 1 to the end of inputs to mult against the bias weights in the weight list
            # print("len of input list: " + str(len(self.prev.outputs)) + "    len of wgt list: " + str(len(self.in_weights)))

            # print("inputs: " + str(self.prev.outputs) + "   weights: " + str(self.in_weights))
            # self.zs = np.dot(self.prev.outputs, np.transpose(self.in_weights)) ## dot(self.prev.outputs, self.prev.in_weights)
            # print("outputs (zs): " + str(self.zs))
            # if (self.act == "relu"):
            #     self.outputs = self.relu(self.zs)
            #     print("outputs after act (as): " + str(self.outputs))

    
    # Compute self.in_derivs, assuming 
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays 
    #       of outputs and of labels, and returns an np array 
    #       of dE/da for each output
    #
    # ...also...update all in_derivs, and add to batch_derivs
    def backpropagate(self, err_prime = None, labels = None):
        global dEdzs
        global ws
        if self.prev is None:
            self.in_derivs = None
            print("through all layers")
            return
        elif self.act_prime == "backSoftmax":
            print("backpropping")
            dEdas = -(labels / self.outputs)
            print("dEdas: " + str(dEdas))
            print("zs: " + str(self.zs))
            act_partials = self.backSoftmax(self.outputs, self.zs)
            print("act partials: " + str(act_partials))
            # dEdz
            self.z_derivs = np.dot(dEdas, act_partials)
            dEdzs = list(self.z_derivs)
            ws = self.in_weights
            print("dEdzs: " + str(self.z_derivs))
            # dEdw
            self.in_derivs = np.outer(self.prev.outputs, self.z_derivs)
            print("dEdws: " + str(self.in_derivs))
        elif self.act_prime == "backRelu":
            print("backpropping")
            dadzs = self.backRelu(self.outputs)
            print("dadzs: " + str(dadzs))
            # sum weights going to next lyr * dEdzs along their lines
            # dEdzs.append(0)
            del ws[0][-1]
            del ws[1][-1]
            del ws[2][-1]
            print("weights: " + str(ws) + " and dEdzs: " + str(dEdzs))
            chain = np.dot(dEdzs, ws)
            print("chain: " + str(chain))
            self.z_derivs = dadzs * chain
            print ("ds: " + str(self.z_derivs))
            self.in_derivs = np.outer(self.prev.outputs, self.z_derivs)
            print("dEdws: " + str(self.in_derivs))

            
        self.batch_derivs = sum(sum(self.in_derivs))
        print("batch derivs: " + str(self.batch_derivs))



    # Adjust all weights by avg gradient accumulated for 
    #  current batch * -|rate|
    def apply_batch(self, batch_size, rate):
        self.in_weights += self.z_derivs * -(rate) # maybe???
     
    # Reset internal data for start of a new batch
    def start_batch(self):
        self.batch_derivs.fill(0.0) # or fill in_derivs???

    # Add delta to the weight from src node in prior layer
    # to trg node in this layer.
    def tweak_weight(self, src, trg, delta):
        self.in_weights[trg][src].sum(delta) # something like this??

    # Return string description of self for debugging
    def __repr__(self):
        return str(self)

    def relu(self, zs):
        return np.maximum(0, zs)
    
    def softmax(self, zs):
        sumVals = np.exp(zs - np.max(zs))
        return sumVals / sumVals.sum()

    def backRelu(self, ais):
        dadzs = []
        for i in range(len(ais)):
            dadzs.append(1) if ais[i] > 0 else dadzs.append(0)
        return dadzs 

    def backSoftmax(self, ais, zis):
        return [[(ais[0] - np.square(ais[0])), -(ais[1] * ais[0]), -(ais[2] * ais[0])], \
         [-(ais[0] * ais[1]), (ais[1] - np.square(ais[1])), -(ais[2] * ais[1])], \
         [-(ais[0] * ais[2]), -(ais[1] * ais[2]), (ais[2] - np.square(ais[2]))]]
      

class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
    # wgts -- list of one 2-d np.array per layer in arch
    def __init__(self, arch, err, wgts = None):
        self.err = err
        self.outputs = None

        # input layer / create list of layers
        self.lyrs = [Layer(arch[0][0], None, None)]
        # self, dim, prev, act, weights = None):
        # add rest of layers
        for i in range(len(arch) - 1):
            self.lyrs.append(Layer(arch[i + 1][0], self.lyrs[i], arch[i + 1][1], wgts[i]))



    
    # Forward propagate, passing inputs to first layer, and returning outputs
    # of final layer
    def predict(self, inputs):
        self.lyrs[0].propagate(inputs)

        for i in range(len(self.lyrs) - 1):
            self.lyrs[i + 1].propagate()

        return self.lyrs[-1].outputs

        
    # Assuming forward propagation is done, return current error, assuming
    # expected final layer output is |labels|
    def get_err(self, labels):
        term1 = 0
        term2 = 0
        for i in range(len(labels)):
            if labels[i] != 0:
                term1 += (labels[i] * np.log(self.outputs[i]))
                term2 += (labels[i] * np.log(labels[i]))
                # print("term1: " + str(term1))
                # print("term2: " + str(term1))

        return -(term1 - term2)

    
    # Assuming a predict was just done, update all in_derivs, and add to
    # batch_derivs
    def backpropagate(self, labels):
        # (self, err_prime = None, labels = None)

        for i in range(len(self.lyrs)):
            self.lyrs[-(i + 1)].backpropagate(labels = labels)

    
    # Verify all partial derivatives for weights by adding an
    # epsilon value to each weight and rerunning prediction to
    # see if change in error correctly reflects weight change
    def validate_derivs(self, inputs, outputs):
        pass

    
    # Run a batch, assuming |data| holds input/output pairs comprising the 
    # batch. Forward propagate for each input, record error, and backpropagate.
    # At batch end, report average error for the batch, and do a derivative 
    # update.
    def run_batch(self, data, rate):
        pass

            
def main(cmd, cfg_file, data_file):
    print("Loading config and data from json files...")
    cfg = open(cfg_file, "r") # Obtain file contents
    cfgParams = json.load(cfg)

    data = open(data_file, "r") # Obtain file contents
    dataParams = json.load(data)

    net = Network(cfgParams['arch'], cfgParams['err'], cfgParams['wgts'])

    initialInputs = dataParams[0][0]
    # net.outputs = net.predict(initialInputs) # in verify now

    labels = dataParams[0][1]
    # loss = net.get_err(labels) # in verify now

    # print("output from final layer of net: " + str(net.outputs))
    # print("loss of net: " + str(accuracy))

    if (cmd == "verify"):
        # 1.
        # Runs a forward propagation to establish an output value...
        net.outputs = net.predict(initialInputs)
        # ...and corresponding error value.
        loss = net.get_err(labels)
        # 2.
        # Reports the actual output vs the expected output, and the resultant error
        print(str(net.outputs) + "  vs  " + str(labels) + "  for  " + str(loss))

        # 3.
        # Runs a single backpropagation to determine dE/dW values
        net.backpropagate(labels)

        # 4.
        # Systematically, for each weight at each level, adjusts that weight by .01, 
        # reruns the forward propagation to get a new error value, and reports the difference 
        # and the ratio of the difference to the expected difference (.01 * dE/dW for the adjusted weight) as a percentage

        # 5.
        # Returns the weight to its original value.
        

    elif (cmd == "run"):
        # net.run_batch(dataParams, 1)
        pass
    else:
        print("Improper command given.  Try 'verify' or 'run'.")

    # print ("arch = " + str(cfgParams['arch']) + "  Shape of arch: " + str(len(cfgParams['arch'])) + 
    #  "\nerror funct = " + str(cfgParams['err']) + "\nweights = " + str(cfgParams['wgts']) + "\nShape of weights: " + str(len(cfgParams['wgts'])) + str(cfgParams['wgts'][0][0]))

    # Close files
    cfg.close()
    data.close()

    
main(sys.argv[1], sys.argv[2], sys.argv[3])