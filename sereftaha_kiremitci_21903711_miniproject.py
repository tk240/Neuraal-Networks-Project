import h5py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

#########################Q1a####################################3
def rgbtoGray(data1):
    shape=np.shape(data1)
    X=np.zeros([shape[0],1,shape[2],shape[3]])
    X[:,0]=data1[:,0]*0.2126+data1[:,1]*0.7152+data1[:,2]*0.0722
    X=X[:,0,:,:]
    return X

def normalizeIm(X):
    # rangE=[0.1,0.9]
    Y= np.zeros(np.shape(X))
    for i, value in enumerate(X):
        Y[i]=X[i]-np.mean(value)
    std=np.std(X)
    Y=np.clip(Y,std*(-3),std*3)
    Y=(Y - Y.min())/(Y.max() - Y.min())
    Y = 0.1 + Y * 0.8  # map to 0.1 - 0.9
    # Y=np.interp(Y,[std*(-3),std*3],rangE)
    return Y

def normalize(x):
    min_x=x.min()
    max_x=x.max()
    return (x-min_x)/max_x-min_x

def Q1():
    data1= h5py.File("data1.h5", 'r')
    data=data1.get('data')
    data_f=np.array(data)   #data in numpy array for plotting

    data=rgbtoGray(data)
    data=normalizeIm(data)

    data1.close()

    pixel = data.shape[1]
    data_t = np.reshape(data, (data.shape[0], pixel ** 2)) #flatten for the training part


    data_f = data_f.transpose((0, 2, 3, 1))
    rgb_im, ind0 = plt.subplots(10, 20, figsize=(20, 10))
    gray_im, ind1 = plt.subplots(10, 20, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')

    im_number=np.random.randint(0,10240,size=(10,20))

    for i in range(10):
        for j,k in enumerate(im_number[i]):
            ind0[i, j].imshow(data_f[k].astype('float'))
            ind0[i, j].axis("off")

            ind1[i, j].imshow(data[k], cmap='gray')
            ind1[i, j].axis("off")


    rgb_im.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    gray_im.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    rgb_im.savefig("200rgb.png")
    gray_im.savefig("200gray.png")
    plt.close("all")

    plt.figure(figsize=(16,16))
    for i,k in enumerate(im_number):    
        plt.subplot(10,20,i+1)
        plt.axis('off')
        plt.imshow(data[k[0]],cmap='gray')
    plt.figure(figsize=(16,16))

    for i,k in enumerate(im_number):  
        plt.subplot(10,20,i+1)
        plt.axis('off')
        plt.imshow(data_f[k[0]])
    ###########################Q1a##################################################
    #Determined by expeirment
    lr = 0.08
    epoch=75
    rho = 0.15
    beta = 2.12
    
    #Q1b these are determined by the manual
    Lambda = 5e-4
    Lin = data.shape[2]*data.shape[1]
    Lhid = 64
    
    params = {"rho": rho, "beta": beta, "Lambda": Lambda, "Lin": Lin, "Lhid": Lhid}
    ae = AutoEncoder(params,data_t)
    W, Js0 = ae.train(data_t,lr, epoch, batchS = 16)
    W1,_,_,_ = W
    Wnorm = normalize(W1)
    fig = plt.figure(figsize=(100, 70))
    for i in range(64):
        fig.add_subplot(8,8,i+1)
        plt.imshow(Wnorm[:,i].reshape((16,16)),cmap='gray')
    fig.savefig("first.png")
        


    ######### low-med-high
    # low med high
    cvJs=[]
    index=0
    Lambda = [0, 4e-4, 15e-4]
    Lhid = [36, 49, 81]
    for l in Lambda:
        for h in Lhid:
            s = np.sqrt(h)
            s = int(s)
            print('λ='+str(l)+ ', L='+str(h))
            params = {"rho": rho, "beta": beta, "Lambda": l, "Lin": Lin, "Lhid": h}
            ae = AutoEncoder(params,data_t)
            W, Js = ae.train(data_t,lr,  epoch, batchS = 16)
            cvJs.append(Js)
            W1,_,_,_ = W
            Wnorm = normalize(W1)
            plt.figure()
            fig = plt.figure(figsize=(100, 70))
            fig.suptitle("lambda = {}, Lhid = {}".format(l, h), fontsize=30)
            for i in range(h):
                fig.add_subplot(s,s,i+1)
                plt.imshow(Wnorm[:,i].reshape((16,16)),cmap='gray')
            fig.savefig("first"+str(index)+".png")
            index+=1
    Lhids = [36, 49, 81]
    lambdas=[0, 4e-4, 15e-4]
    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Cross Validation Loss", fontsize=20)
    plt.plot(Js0, label='λ='+str(5e-4)+' L=64')
    a=0
    for k in lambdas:
        for i in Lhids:
#            for j in cvJs:
            plt.plot(cvJs[a], label='λ='+str(k)+', L='+str(i))
            plt.legend()
            plt.title("")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            a+=1
    plt.savefig("CVLost.png")

######################  End of Q1  #################################################

#########################Q2#########################################################
def q2():
    dir_= 'data2.h5'
    data2= h5py.File(dir_, 'r')
    testd=np.array(data2.get('testd'))
    testx=np.array(data2.get('testx'))
    traind=np.array(data2.get('traind'))
    trainx=np.array(data2.get('trainx'))
    vald=np.array(data2.get('vald'))
    valx=np.array(data2.get('valx'))
    words=np.array(data2.get('words'))

    data2.close()

    traind = np.reshape(traind, (traind.shape[0], 1))
    vald = np.reshape(vald, (vald.shape[0], 1))
    testd = np.reshape(testd, (testd.shape[0], 1))
    words = np.reshape(words, (words.shape[0], 1))

    ########## Part a ###############

    lr = 0.15
    mom = 0.85
    epoch = 50 
    batchS = 200
  
    # 8 , 64 ##################
    d2,p2=[8,64]
    sizeL2=[750, d2,p2,250]
    lenL = len(sizeL2) - 1


    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Train and Validation Lost", fontsize=20)

    nlpnet2 = nlp(sizeL2, lenL)
    out2 = nlpnet2.train(trainx, traind, valx, vald, lr, epoch, batchS, mom)
    lostTrainL2, lostValL2, accTrainL2, accValL2 = out2.values()
    

    plt.plot(lostTrainL2, "C0", label="Train Loss")
    plt.plot(lostValL2, "C3", label="Validation Loss")
    plt.legend()
    plt.title("(D, P) = 8, 64")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.savefig("question2Lost.png")


    # 16, 128 ################
    d1,p1=[16,128]
    sizeL1=[750, d1,p1,250]

    lenL = len(sizeL1) - 1

    nlpnet1 = nlp(sizeL1, lenL)
    out1 = nlpnet1.train(trainx, traind, valx, vald, lr, epoch, batchS, mom)
    lostTrainL1, lostValL1, accTrainL1, accValL1 = out1.values()

    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Train and Validation Lost", fontsize=20)

    #plt.subplot(1, 3, 2)
    plt.plot(lostTrainL1, "C0", label="Train Loss")
    plt.plot(lostValL1, "C3", label="Validation Loss")
    plt.legend()
    plt.title("(D, P) = 16, 128")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.savefig("question2Lost2.png")

    # 32, 256 ###########
    d,p=[32,256]
    sizeL=[750, d,p,250]

    lenL = len(sizeL1) - 1


    nlpnet = nlp(sizeL, lenL)
    out = nlpnet.train(trainx, traind, valx, vald, lr, epoch, batchS, mom)
    lostTrainL, lostValL, accTrainL, accValL = out.values()

    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Train and Validation Lost", fontsize=20)

    plt.plot(lostTrainL, "C0", label="Train Loss")
    plt.plot(lostValL, "C3", label="Validation Loss")
    plt.legend()
    plt.title("(D, P) = (32, 256)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")

    plt.savefig("question2Lost3.png")
    
    
    
    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Train and Validation Accuracy", fontsize=20)
    
    plt.plot(accTrainL2, "C0", label="Train Accuracy")
    plt.plot(accValL2, "C3", label="Validation Accuracy")
    plt.legend()
    plt.title("(D, P) = (8, 64)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("question2Accuracy.png")
    
    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Train and Validation Accuracy", fontsize=20)
    
    plt.plot(accTrainL1, "C0", label="Train Accuracy")
    plt.plot(accValL1, "C3", label="Validation Accuracy")
    plt.legend()
    plt.title("(D, P) = (16, 128)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("question2Accuracy2.png")
    
    fig = plt.figure(figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("Train and Validation Accuracy", fontsize=20)
    
    plt.plot(accTrainL, "C0", label="Train Accuracy")
    plt.plot(accValL, "C3", label="Validation Accuracy")
    plt.legend()
    plt.title("(D, P) = (32, 256)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.savefig("question2Accuracy3.png")
    
    

    ######### b  ###################
    temp = testx - 1
    data0 = np.zeros((temp.shape[0], 0))
    for i in range(temp.shape[1]):
        temp2 = np.zeros((temp.shape[0], 250))
        temp2[np.arange(temp.shape[0]), temp[:, i]] = 1
        data0 = np.hstack((data0, temp2))


    testClas,_ = nlpnet.choice(data0)

    print("\n\nTest Accuracy of (32, 256): ", (testClas == testd).mean() )

    ind = 10  
    np.random.seed(666)
    p = np.random.permutation(testx.shape[0])
    testx = testx[p][:ind]
    testd = testd[p][:ind]
    ##################################################
    temp = testx - 1
    data1 = np.zeros((temp.shape[0], 0))
    for i in range(temp.shape[1]):
        temp2 = np.zeros((temp.shape[0], 250))
        temp2[np.arange(temp.shape[0]), temp[:, i]] = 1
        data1 = np.hstack((data1, temp2))

    _,predTest = nlpnet.choice(data1)

    n = 10
    s = (np.argsort(-predTest, axis=1) + 1)[:, :n]

    for i in range(ind):
            print("\n")
            print("Sequence:", words[testx[i][0] - 1], words[testx[i][1] - 1], words[testx[i][2] - 1])
            print("Label:", words[testd[i] - 1])
            for j in range(n):
                print(str(j + 1) + ". ", words[s[i][j] - 1])
#########################End of Q2 #################################################

##################### Q3############################################################
def q3():
    filename = "data3.h5"
    h5 = h5py.File(filename, 'r')
    trX = h5['trX'][()].astype('float64')
    tstX = h5['tstX'][()].astype('float64')
    trY = h5['trY'][()].astype('float64')
    tstY = h5['tstY'][()].astype('float64')
    h5.close()

    mom = 0.85
    lr = 0.1
    epoch = 50 #max 50
    batch_size = 32
    
    hs = [64,16]
    print("RNN\n")
    rnnQ1 = sRNN(128,hs, modelType='rnn')
    lostTrain, lostVal, accTrain, accVal, trainconf = rnnQ1.train(trX, trY, lr, mom, batch_size, epoch, modelType='rnn').values()
    accuracy, _, confidence = rnnQ1.choice(tstX, tstY)

    print("\nTest Accuracy: ", accuracy, "\n")

# =============================================================================
    hs2 = [16,32]
    mom = 0.85
    lr = 0.1
    epoch = 50 #max50
    batch_size = 32
    print("LSTM\n")
    lstmQ2 = sRNN(128, hs=hs2, modelType="lstm")
    lostTrain2, lostVal2, accTrain2, accVal2, trainconf2 = lstmQ2.train(trX, trY, lr, mom, batch_size,epoch,modelType='lstm').values()
    accuracy2,_,confidence2 = lstmQ2.choice(tstX, tstY)
    print("\nTest Accuracy: ", accuracy2, "\n")
    
    hs3 = [64,32]
    mom = 0.85
    lr = 0.07
    epoch = 50 #50
    batch_size = 32
    print("GRU\n")
    gruQ3 = sRNN(128, hs=hs3, modelType="gru")
    lostTrain3, lostVal3, accTrain3, accVal3, trainconf3 = gruQ3.train(trX, trY, lr, mom, batch_size,epoch,modelType='gru').values()
    accuracy3,_,confidence3 = gruQ3.choice(tstX, tstY)
    print("\nTest Accuracy: ", accuracy3, "\n")
    
# =============================================================================
    #plot the graphs
    fig = plt.figure(figsize=(30, 15))
    fig.suptitle("RNN\n Train Accuracy: {:.1f} -- Validation Accuracy: {:.1f} -- Test Accuracy: {:.1f}\n "
    .format(accTrain[-1], accVal[-1], accuracy), fontsize=20)
    plt.subplot(1, 2, 1)
    plt.plot(lostTrain, "C2")
    
    plt.title("Cross-Entropy RNN Loss")
    plt.xlabel("Epoch num")
    plt.ylabel("Loss")
    plt.plot(lostVal, "C3")

    plt.subplot(1, 2, 2)
    plt.plot(accTrain, "C2")
    
    
    plt.title("RNN Accuracy")
    plt.xlabel("Epoch num")
    plt.ylabel("Accuracy")
    plt.plot(accVal, "C3")
    
    plt.savefig("question3RNNacc.png")

    plt.figure(figsize=(20, 10))

    
    lbls = [1, 2, 3, 4, 5, 6]
    plt.subplot(1, 2, 1)
    sn.heatmap(trainconf, annot=True, annot_kws={"size": 8}, xticklabels=lbls, yticklabels=lbls,
    cmap=sn.cm.rocket, fmt='g')
    
    plt.title("Confusion Matrix of Training")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")
    plt.subplot(1, 2, 2)
    sn.heatmap(confidence, annot=True, annot_kws={"size": 8}, xticklabels=lbls, yticklabels=lbls,
    cmap=sn.cm.rocket, fmt='g')
    
    plt.title("Confusion Matrix of Test")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")
    plt.savefig("question3ConfRec.png")
    
    
# =============================================================================
    ###
    fig = plt.figure(figsize=(30, 15))
    fig.suptitle("LSTM\n Train Accuracy: {:.1f} -- Validation Accuracy: {:.1f} -- Test Accuracy: {:.1f}\n "
    .format( accTrain2[-1], accVal2[-1], accuracy2), fontsize=20)
    plt.subplot(1, 2, 1)
    plt.plot(lostTrain2, "C2", label="Train")
    plt.legend()
    
    plt.title("Cross-Entropy LSTM Loss")
    plt.xlabel("Epoch num")
    plt.ylabel("Loss")
    plt.plot(lostVal2, "C3", label="Validation")
    plt.legend()
    

    plt.subplot(1, 2, 2)
    plt.plot(accTrain2, "C2",label="Train")
    plt.legend()

    
    plt.title("Training Accuracy")
    plt.xlabel("Epoch num")
    plt.ylabel("Accuracy")
    plt.plot(accVal2, "C3", label="Validation")
    plt.legend()
   
    plt.savefig("question3LSTM.png")
    plt.figure(figsize=(20, 10))
    
    lbls = [1, 2, 3, 4, 5, 6]
    plt.subplot(1, 2, 1)
    sn.heatmap(trainconf2, annot=True, annot_kws={"size": 8}, xticklabels=lbls, yticklabels=lbls,
    cmap=sn.cm.rocket, fmt='g')
    
    plt.title("Confusion Matrix of Training")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")
    plt.subplot(1, 2, 2)
    sn.heatmap(confidence2, annot=True, annot_kws={"size": 8}, xticklabels=lbls, yticklabels=lbls,
    cmap=sn.cm.rocket, fmt='g')
    plt.title("Confusion Matrix of Test")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")
    plt.savefig("question3ConfLSTM.png")

   
   
    ###
    fig = plt.figure(figsize=(30, 15))
    fig.suptitle("GRU\nTrain Accuracy: {:.1f} -- Validation Accuracy: {:.1f} -- Test Accuracy: {:.1f}\n "
    .format( accTrain3[-1], accVal3[-1], accuracy3), fontsize=20)
    plt.subplot(1, 2, 1)
    plt.plot(lostTrain3, "C2", label='Train')
    plt.legend()
    
    plt.title("Cross-Entropy Loss")
    plt.xlabel("Epoch num")
    plt.ylabel("Loss")
    plt.plot(lostVal3, "C3", label='Validation')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accTrain3, "C2", label='Train')
    plt.legend()
    
    plt.title(" Accuracy")
    plt.xlabel("Epoch num")
    plt.ylabel("Accuracy")
    plt.plot(accVal3, "C3", label='Validation')
    plt.legend()
    

    plt.savefig("question3GRU.png")
    plt.figure(figsize=(20, 10))
    
    lbls = [1, 2, 3, 4, 5, 6]
    plt.subplot(1, 2, 1)
    sn.heatmap(trainconf3, annot=True, annot_kws={"size": 8}, xticklabels=lbls, yticklabels=lbls,
    cmap=sn.cm.rocket, fmt='g')
    
    plt.title('Confusion Matrix of Training')
    plt.ylabel("Actual")
    plt.xlabel("Prediction")
    plt.subplot(1, 2, 2)
    sn.heatmap(confidence3, annot=True, annot_kws={"size": 8}, xticklabels=lbls, yticklabels=lbls,
    cmap=sn.cm.rocket, fmt='g')
    
    plt.title("Confusion Matrix of Testing")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")
    plt.savefig("question3ConfGRU.png")
# =============================================================================

########################End of Q3###################################################

##########Q1 Calss###############
class AutoEncoder(object):
    def __init__(self,params,data):
        Lin = params["Lin"]
        Lhid = params["Lhid"]        
        self.rho = params["rho"]
        self.beta = params["beta"]
        self.Lambda = params["Lambda"]


        Lout = Lin # autoencoder takes like this
        L = Lin + Lhid
        
        w_0 = np.sqrt(6/L)
        W_1 = np.random.uniform(-1*w_0,w_0,size=(Lin, Lhid))
        b_1 = np.random.uniform(-1*w_0,w_0,size=(1,Lhid))

        W_2 = W_1.T
        b_2 = np.random.uniform(-1*w_0,w_0,size=(1, Lout))

        self.We = [W_1, W_2, b_1, b_2]

    def aeCost(self,We,data):
        W_1,W_2,b_1,b_2=We
        N=data.shape[0]

        update=np.dot(data,W_1)+b_1
        result=self.sigmoid(update)
        update2=np.dot(result,W_2)+b_2
        output=self.sigmoid(update2)

        dloss = -(data-output)/N
        dtykh_1 = self.Lambda*W_1
        dtykh_2 = self.Lambda*W_2

        #the coss function in three parts (instructed in the manual)

        loss = 0.5/N * (np.linalg.norm(data - output, axis=1) ** 2).sum()
        tykh = 0.5 * self.Lambda * (np.sum(W_1 ** 2) + np.sum(W_2 ** 2))
        rho_hat = result.mean(axis=0, keepdims=True)
        KLdiv = self.rho * np.log(self.rho/rho_hat) + (1 - self.rho) * np.log((1 - self.rho)/(1 - rho_hat))
        KLdiv= self.beta*KLdiv.sum()

        dKLdiv = self.beta * (- self.rho/rho_hat + (1-self.rho)/(1 - rho_hat))/N
        
        #sum of three parts is the loss
        J=loss+tykh+KLdiv
        cache={'data':data,'result':result,'output':output}
        J_grad={'dloss':dloss, 'dtykh_1':dtykh_1, 'dtykh_2': dtykh_2, 'dKLdiv':dKLdiv}
        
        return J_grad,cache,J

    def train(self,data,learning_rate,epoch,batchS):
        N=data.shape[0]
        Js=[]
        iteration = round(N/batchS)
        We=self.We
        for i in range(epoch):
            totJ = 0
            indS = 0
            indE = batchS

            randomChoice = np.random.permutation(N)
            randomData = data[randomChoice]

            mWe = (0, 0, 0, 0)

            for j in range(iteration):

                batch = randomData[indS:indE]

                J_grad, cache, J = self.aeCost(We,batch)
                We=self.GDsolver(We, J_grad, cache, learning_rate)

                totJ = totJ+ J
                indS = indE
                indE += batchS

            totJ /=iteration
            print("Loss: {:.2f}, Epoch {} out of {}".format(totJ, i+1, epoch))
            Js.append(totJ)
        print("\n")
        return We,Js

    def learn(self,learning_rate,grads,We):        
        grads = grads*learning_rate
        We[0]-=grads[0]
        We[1]-=grads[1]
        We[2]-=grads[2]
        We[3]-=grads[3]
        return We
        
    def GDsolver(self,We,J_grad,cache,learning_rate):
        # dW_1,dW_2,db_1,db_2= [0, 0, 0, 0]
        data=cache['data']
        hidden=cache['result']
        dHidden=hidden-hidden**2
        output=cache['output']
        dOutput=output-output**2

        dloss=J_grad['dloss']
        dtykh_1=J_grad['dtykh_1']
        dtykh_2=J_grad['dtykh_2']
        dKLdiv=J_grad['dKLdiv']

        change=dloss*dOutput

        W_2=np.dot(hidden.T,change)+dtykh_2
        db_2=np.sum(change,axis=0,keepdims=True)

        change1 = dHidden * (np.dot(change,We[1].T) + dKLdiv)

        W_1 = np.dot(data.T, change1) + dtykh_1
        db_1 = np.sum(change1,axis=0, keepdims=True) 

        dw_2=(1/2)*(W_1.T + W_2)
        dw_1= W_2.T

        grads=np.array([dw_1,dw_2,db_1,db_2])
        We=self.learn(learning_rate,grads,We)
        return We

    def softmax(self,x):
        soft=np.zeros(1,len(x))
        for i in len(x):
            soft[i]=np.exp(x[i])/sum(np.exp(x))
        return soft

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def relu(self,X):
        return np.maximum(0,X)
##########Q2 Class################
class nlp(object):
    def __init__(self, sizeL, lenL = 2, std = 0.01, seed=666):
        #sizeL= all layer len
        # lenL=layer length, 
        self.seed = seed
        Ws = []
        bs= []

        for i in range(lenL):
            if i==0:
                np.random.seed(self.seed)
                a = np.random.normal(0, std, size=(int(sizeL[i]/3), sizeL[i + 1]))
                Ws.append(np.vstack((a, a, a)))
                np.random.seed(self.seed)
                bs.append(np.zeros((1, sizeL[i + 1])))
                continue
            np.random.seed(self.seed)
            Ws.append(np.random.normal(0, std, size=(sizeL[i], sizeL[i + 1])))
            np.random.seed(self.seed)
            bs.append(np.random.normal(0, std, size=(1, sizeL[i + 1])))

        self.mom = {"Ws": [None] * lenL, "bs": [None] * lenL}
        self.sizeL = sizeL
        self.param = {"Ws": Ws, "bs": bs}
        self.lenL = lenL
        self.activ = ["sigmoid"] * (lenL - 1) + ["softmax"]

    def cost(self, data, gt):
        param=self.param
        Ws,bs =param.values()
        activ = self.activ
        dataL = [data]
        grads = [1]
        batchS = data.shape[0]
        lenL=self.lenL

        for i in range(lenL):
            post = np.dot(dataL[i], Ws[i]) + bs[i]
            out, grad = self.activition( post,activ[i])
            dataL.append(out)
            grads.append(grad)

        guess = dataL[-1]
        cost = self.CE(gt, guess)
        dE = guess
        dE[gt == 1]-= 1
        dE = dE/batchS

        dWs = []
        dbs = []
        identity = np.ones((1, batchS))

        for i in reversed(range(lenL)):
            dWs.append(np.dot(dataL[i].T , dE))
            dbs.append(np.dot(identity , dE))
            dE = grads[i] * (np.dot(dE , Ws[i].T))

        return cost, {'dWs': dWs[::-1], 'dbs': dbs[::-1]}

    def CE(self, expect, result):
        log=np.log(result)
        return np.sum(-1*expect* log)/ expect.shape[0]
    
    def activition(self,data,activation):
        if activation=='softmax':
            Z=np.exp(data)/np.sum(np.exp(data),axis=1,keepdims=True)
            d=None
        elif activation=='sigmoid':
            Z=1/(1+np.exp(-data))
            d=Z-Z**2
        elif activation=='relu':
            #Z=np.max(0.0,data)
            Z = data * (data > 0)
            d=1*(data>0)
        elif activation=='tanh':
            Z=np.tanh(data)
            d=1- Z**2
        return Z,d

    def choice(self, data):
        Ws = self.param["Ws"]
        bs = self.param["bs"]
        activ = self.activ
        lenL=self.lenL
        dataL = [data]

        for i in range(lenL):
            post = np.dot(dataL[i], Ws[i]) + bs[i]
            dataL.append(self.activition(post,activ[i])[0])
        # three word embedding one for each word
        out = (np.argmax(dataL[-1], axis=1) + 1).T
        out = np.reshape(out, (out.shape[0], 1))
        return out,dataL[-1]

    def train(self, data, gt, valData, valGt, lr=0.2, epoch=50,batchS=100,mom=0):
        lostTrainL = []
        lostValL = []
        accTrainL = []
        accValL = []
    
        param=self.param
        iteration = int(data.shape[0] / batchS)
        lenL=self.lenL
        sizeL=self.sizeL
        moms=self.mom

        #one hot encode word out of 250 words so the matrix sghows the index of corresponding word
        temp = data - 1
        data0 = np.zeros((temp.shape[0], 0))
        for i in range(temp.shape[1]):
            temp2 = np.zeros((temp.shape[0], 250))
            temp2[np.arange(temp.shape[0]), temp[:, i]] = 1
            data0 = np.hstack((data0, temp2))

        temp = gt - 1
        gt0 = np.zeros((temp.shape[0], 0))
        for i in range(temp.shape[1]):
            temp2 = np.zeros((temp.shape[0], 250))
            temp2[np.arange(temp.shape[0]), temp[:, i]] = 1
            gt0 = np.hstack((gt0, temp2))

        temp = valData - 1
        valDAta0 = np.zeros((temp.shape[0], 0))
        for i in range(temp.shape[1]):
            temp2 = np.zeros((temp.shape[0], 250))
            temp2[np.arange(temp.shape[0]), temp[:, i]] = 1
            valDAta0 = np.hstack((valDAta0, temp2))
        
        temp = valGt - 1
        valGt0 = np.zeros((temp.shape[0], 0))
        for i in range(temp.shape[1]):
            temp2 = np.zeros((temp.shape[0], 250))
            temp2[np.arange(temp.shape[0]), temp[:, i]] = 1
            valGt0 = np.hstack((valGt0, temp2))

        for i in range(epoch):

            np.random.seed(self.seed)
            shuff = np.random.permutation(data0.shape[0])
            data0 = data0[shuff]
            gt0 = gt0[shuff]
            gt = gt[shuff]

            for e in range(lenL):
                self.mom["Ws"][e] = np.zeros((sizeL[e],sizeL[e+1]))
                self.mom["bs"][e] = np.zeros((1, sizeL[e+1]))

            first = 0
            last = batchS
            lostTrain = 0

            for j in range(iteration):
                batchTr0 = data0[first:last]
                batchGt0 = gt0[first:last]
                batchGt = gt[first:last]

                cost, grad = self.cost(batchTr0, batchGt0)
                lostTrain += cost

                for k in range(lenL):

                    if k == 0:
                        moms["Ws"][k] = lr * (grad["dWs"][k] + mom * moms["Ws"][k])
                        dword1, dword2, dword3 = np.array_split(moms["Ws"][k], 3, axis=0)
                        dWord = (dword1 + dword2 + dword3)/3
                        param["Ws"][k] -= np.vstack((dWord, dWord, dWord))
                       

                    moms["Ws"][k] = lr * (grad["dWs"][k] + mom * moms["Ws"][k])
                    moms["bs"][k] = lr * (grad["dbs"][k] + mom * moms["bs"][k])
                    param["Ws"][k] = param["Ws"][k]-moms["Ws"][k]
                    param["bs"][k] = param["bs"][k]-moms["bs"][k]

                first = last
                last += batchS
            
            choiceT,_ = self.choice(data0)
            accuracyTr = (choiceT == gt).mean() 

            choiceV,_ = self.choice(valDAta0)
            accuracyV = (choiceV == valGt).mean()

            _,val_pred = self.choice(valDAta0)
            val_loss = self.CE(valGt0, val_pred)

            
            print('\r(D, P) = (%d, %d). loos Train: %f, loss Val: %f, accuracy Train: %f, accuracy Val: %f [%d out of %d].'
                    % (sizeL[1], sizeL[2], lostTrain/(j+1), val_loss, accuracyTr, accuracyV, i + 1, epoch))

            lostTrainL.append(lostTrain/iteration)
            lostValL.append(val_loss)
            accTrainL.append(accuracyTr)
            accValL.append(accuracyV)


            if i > 15 :
                check = lostValL[-15:]
                check = sum(check) / len(check)

                lim = 0.02

                if (check - lim) < val_loss < (check + lim) and val_loss < 3.5:
                    print(" Training is stopped due to cross entropy convergence in Validation ")
                    return {'lostTrainL': lostTrainL, 'lostValL': lostValL,
                            'accTrainL': accTrainL, 'accValL': accValL}
        return {'lostTrainL': lostTrainL, 'lostValL': lostValL,
                            'accTrainL': accTrainL, 'accValL': accValL}
###########End################
##########Q3 Class###############
class sRNN(object):

    def __init__(self, n_number,hs, modelType ,outNum=6, inputs=3):
        #train=(3000, 150,3) sample,series,sensor
        #neuron number= 128 
        self.modelType0=modelType
        trainS=([inputs,n_number]+hs)+[outNum]
        self.Size=trainS
        self.Layer_len=len(trainS)-1

        hs=trainS[1]

        if modelType=='rnn':
            n=inputs+hs
            x=np.sqrt(6/n)
            W_ih=np.random.uniform(-x, x, size=(inputs, hs))
            x = np.sqrt(6 / (2*hs))
            W_hh = np.random.uniform(-x, x, size=(hs, hs))
            b = np.zeros((1, hs))
            self.We = {"Wih": W_ih, "Whh": W_hh, "b": b}
            self.inLayer = {"Wih": 0, "Whh": 0, "b": 0}

        elif modelType=='lstm':
            n=inputs+2*hs
            X=np.sqrt(6/n)
            Wini = np.random.uniform(-X, X, size=(n-hs, hs))
            Win = np.random.uniform(-X, X, size=(n-hs, hs))
            Wsec = np.random.uniform(-X, X, size=(n-hs, hs))
            Wout = np.random.uniform(-X, X, size=(n-hs, hs))

            bini = np.zeros((1, hs))
            bin1 = np.zeros((1, hs))
            bsec = np.zeros((1, hs))
            bout = np.zeros((1, hs))            
            
            self.We = {"Wini": Wini, "bini": bini,
                      "Win": Win, "bin1": bin1,
                      "Wsec": Wsec, "bsec": bsec,
                      "Wout": Wout, "bout": bout} 
            self.inLayer = {"Wini": 0, "bini": 0,
                            "Win": 0, "bin1": 0,
                            "Wsec": 0, "bsec": 0,
                            "Wout": 0, "bout": 0}
        elif modelType=='gru':
            Nih = np.sqrt(6 / (inputs + hs))
            Nhh = np.sqrt(6 / (2* hs))

            Wa = np.random.uniform(-Nih, Nih, size=(inputs, hs))
            Ta = np.random.uniform(-Nhh, Nhh, size=(hs, hs))
            ba = np.zeros((1, hs))

            Wb = np.random.uniform(-Nih, Nih, size=(inputs, hs))
            Tb = np.random.uniform(-Nhh, Nhh, size=(hs, hs))
            bb = np.zeros((1, hs))

            Wc = np.random.uniform(-Nih, Nih, size=(inputs, hs))
            Tc = np.random.uniform(-Nhh, Nhh, size=(hs, hs))
            bc = np.zeros((1, hs))

            self.We = {"Wa": Wa, "Ta": Ta, "ba": ba,
                      "Wb": Wb, "Tb": Tb, "bb": bb,
                      "Wc": Wc, "Tc": Tc, "bc": bc}       
            self.inLayer = {"Wa": 0, "Ta": 0, "ba": 0,
                            "Wb": 0, "Tb": 0, "bb": 0,
                            "Wc": 0, "Tc": 0, "bc": 0}

        wMuLP = []
        bMuLP = []
        for i in range(1, self.Layer_len):
            x = np.sqrt(6 / (trainS[i] + trainS[i+1]))
            wMuLP.append(np.random.uniform(-x, x, size=(trainS[i], trainS[i+1])))
            bMuLP.append(np.zeros((1, trainS[i+1])))
        self.MLPlayer_len = len(wMuLP)
        self.MLPparam = {"W": wMuLP, "b": bMuLP}
        self.MLPm = {"W": [0] * self.MLPlayer_len, "b": [0] * self.MLPlayer_len,}
        
    def forward(self,data, modelType):
        MLPparam=self.MLPparam
        out=[]
        d=[]
        hidden=0
        dHidden=0
        cache=0

        modelType0=self.modelType0
        
        if modelType0 =='gru':
            hidden,cache=self.forwardGRU(data)
            out.append(hidden) 
            d.append(1)
        elif modelType0 =='lstm':
            hidden,cache=self.forwardLSTM(data)
            out.append(hidden) 
            d.append(1)
        elif modelType0 =='rnn':
            hidden,dHidden=self.forwardRNN(data)
            out.append(hidden[:, -1, :]) 
            d.append(dHidden[:, -1, :])
            
        for i in range(self.MLPlayer_len-1):
            activition, deriv = self.forwardMLP(out[-1], MLPparam["W"][i], MLPparam["b"][i], "relu")
            out.append(activition)
            d.append(deriv)

        choice = self.forwardMLP(out[-1], MLPparam["W"][-1], MLPparam["b"][-1], "softmax")[0]
        return choice,out,d,hidden,dHidden,cache

    def backward(self,data,activition,dMLP,dEp, modelType,hidden=None,dhidden=None,cache=None):
        MLPparam=self.MLPparam
        MLPlayer_len=self.MLPlayer_len
        gMLP ={"W": [0] * MLPlayer_len, "b": [0] * MLPlayer_len}
        # back before rnn
        for i in reversed(range(MLPlayer_len)):
            gMLP["W"][i], gMLP["b"][i], dEp = self.backMLP(MLPparam["W"][i], activition[i], dMLP[i], dEp)
        # back in time
        if modelType == 'gru':
            gRNN = self.backGRU(data, cache, dEp)
        elif modelType == 'lstm':
            gRNN = self.backLSTM(cache, dEp)
        elif modelType == 'rnn':
            gRNN = self.backRNN(data,hidden, dhidden, dEp)

        return gRNN, gMLP

    def forwardMLP(self,data,W,b,activition):
        MLP=np.dot(data,W)+b
        Z,d=self.activition(MLP,activition)
        return Z,d

    def backMLP(self, w, preout,dPreout,dEpre):
        db=np.sum(dEpre,axis=0,keepdims=True)
        dW=np.dot(preout.T,dEpre)
        dE=dPreout*np.dot(dEpre,w.T) #new delta from old delta and derivative
        return dW,db,dE

    def forwardRNN(self,data):
        numN=self.Size[1] #neuron number
        dims=np.shape(data)
        numS=dims[0] #sample number
        t=dims[1]    #time sequence
        sensor=dims[2]  #three sensor dim
        
        We=self.We
        Wih=We['Wih']
        Whh=We['Whh']
        b=We['b']

        hiddenL=np.zeros((numS,numN))
        hidden=np.empty((numS,t,numN))
        dHidden=np.copy(hidden)

        for i in range(t):
            d=data[:,i,:]
            a=np.dot(d,Wih)
            b=np.dot(hiddenL,Whh)+b
            hidden[:, i, :],dHidden[:, i, :] = self.activition(a+b, "tanh")
            hiddenL = hidden[:, i, :]
        return hidden,dHidden
            
    def backRNN(self,data,hidden,dhidden,dEp):
        numN=self.Size[1] #neuron number
        dims=np.shape(data)
        numS=dims[0] #sample number
        t=dims[1]    #time sequence
        sensor=dims[2]  #three sensor dim
        
        We=self.We
        Whh=We['Whh']
        dW_ih,dW_hh,db=[0,0,0]

        for i in reversed(range(t)):
            batch=data[:,i,:]

            if i>0:
                hiddenL=hidden[:,i-1,:]
                dhiddenL=dhidden[:,i-1,:]
            else:
                hiddenL=np.zeros((numS,numN))  #for first layer
                dhiddenL=0
            dW_ih=dW_ih+np.dot(batch.T,dEp)
            dW_hh=dW_hh+np.dot(hiddenL.T,dEp)
            db=db+np.sum(dEp,axis=0,keepdims=True)
            dEp= np.dot(dEp,Whh)*dhiddenL
        grads={'dWih':dW_ih,'dWhh':dW_hh,
                'db':db, 'dEp':dEp}
        return grads

    def forwardLSTM(self,data):
        #store activation
        We=self.We
        Wini,bini,Win,bin1,Wsec,bsec,Wout,bout=We.values()
        numN=self.Size[1] #neuron number
        dims=np.shape(data)
        numS=dims[0] #sample number
        t=dims[1]    #time sequence
        sensor=dims[2]  #three sensor dim

        hiddenL = np.zeros((numS, numN))
        cL = np.zeros((numS, numN)) #for firrst iter

        k = np.empty((numS, t, sensor + numN))
        c = np.empty((numS, t, numN))

        hiddenf = np.empty((numS, t, numN))
        hiddeni = np.empty((numS, t, numN))
        hiddenz = np.empty((numS, t, numN))
        hiddeno = np.empty((numS, t, numN))
        tanhc = np.empty((numS, t, numN))
        
        dhiddenf = np.empty((numS, t, numN))
        dhiddeni= np.empty((numS, t, numN))
        dhiddenz = np.empty((numS, t, numN))
        dhiddeno = np.empty((numS, t, numN))
        dtanh = np.empty((numS, t, numN))

        for i in range(t):
            dataTime=data[:,i,:]
            temp=np.hstack((hiddenL,dataTime))
            f,df = self.activition(np.dot(temp, Wini) + bini, "sigmoid")
            i0,di = self.activition(np.dot(temp, Win) + bin1, "sigmoid")
            z,dz = self.activition(np.dot(temp ,Wsec) + bsec, "tanh")
            o,do= self.activition(np.dot(temp, Wout) + bout, "sigmoid")          

            a = np.multiply(z , i0)
            b= np.multiply(cL , f)
            multC= a+b
            tanh_c,dtan_h=self.activition(multC,'tanh')
            multZ=np.multiply(o,tanh_c)

            cL=multC
            hiddenL=multZ
           

            hiddenf[:,i,:],dhiddenf[:,i,:] = f,df
            hiddeni[:,i,:],dhiddeni[:,i,:] = i0,di
            hiddenz[:,i,:],dhiddenz[:,i,:] =z,dz 
            hiddeno[:,i,:],dhiddeno[:,i,:] =o,do
            tanhc[:,i,:],dtanh[:,i,:] = tanh_c,dtan_h
            k[:,i,:]=temp

        return multZ, {'k':k,'c':c,'hiddenf':hiddenf,'hiddeni':hiddeni,'hiddenz':hiddenz,'hiddeno':hiddeno,
                        'dhiddenf':dhiddenf,'dhiddeni':dhiddeni,'dhiddenz':dhiddenz,'dhiddeno':dhiddeno,
                        'tanhc':tanhc,'dtanh':dtanh}

    def backLSTM(self,cache,dEu):
        We=self.We
        Wini,_,Win,_,Wsec,_,Wout,_=We.values()
        k,c,hiddenf,hiddeni,hiddenz,hiddeno,dhiddenf,dhiddeni,dhiddenz,dhiddeno,tanhc,dtanh = cache.values()
        numN=self.Size[1] #neuron number
        t=np.shape(k)[1]
        
        dWini,dbini,dWin,dbin1,dWsec,dbsec,dWout,dbout=0,0,0,0,0,0,0,0
        #gradients start with zero since no grad first

        for r in reversed(range(t)):
            know = k[:, r, :]

            if r>0:
                cLast=c[:,r-1,:]
            elif r<=0:
                cLast=0 #0  for the ebeginning

            delc = dEu * hiddeno[:, r, :] * dtanh[:, r, :]
            delhf = delc * cLast * dhiddenf[:, r, :]
            delhi = delc * hiddenz[:, r, :] * dhiddeni[:, r, :]
            delhz = delc * hiddeni[:, r, :] * dhiddenz[:, r, :]
            delho = dEu * tanhc[:, r, :] * dhiddeno[:, r, :]
            
            #add to weight's change
            dWini += np.dot(know.T ,delhf)
            dbini += np.sum(delhf,axis=0, keepdims=True)

            dWin += np.dot(know.T , delhi)
            dbin1 += np.sum(delhi,axis=0, keepdims=True)

            dWsec += np.dot(know.T, delhz)
            dbsec += np.sum(delhz,axis=0, keepdims=True)

            dWout +=np.dot(know.T, delho)
            dbout += np.sum(delho,axis=0, keepdims=True)

            #update gradients of gates
            delf=np.dot(delhf, Wini.T[:,:numN])
            deli=np.dot(delhi, Win.T[:,:numN])
            delz=np.dot(delhz, Wsec.T[:,:numN])
            delo=np.dot(delho, Wout.T[:,:numN])

            dEu = delf+deli+delz+delo
        
        return {'dWini':dWini,'dbini':dbini, 'dWin':dWin,'dbin1':dbin1
                ,'dWsec':dWsec,'dbsec':dbsec,'dWout':dWout,'dbout':dbout}

    def forwardGRU(self,data):
        Wa, Ta, ba, Wb, Tb, bb, Wc, Tc, bc=self.We.values()
        numN=self.Size[1] #neuron number
        dims=np.shape(data)
        numS=dims[0] #sample number
        t=dims[1]    #time sequence
        sensor=dims[2]  #three sensor dim 

        hiddenL = np.zeros((numS, numN))
        tmp=(numS,t,numN)
        z = np.empty(tmp)
        dz = np.empty(tmp)
        r = np.empty(tmp)
        dr = np.empty(tmp)
        htil = np.empty(tmp)
        dhtil = np.empty(tmp)
        h = np.empty(tmp)

        for i in range(t):
            dataT = data[:, i, :]
            z[:, i, :], dz[:, i, :] = self.activition(np.dot(dataT, Wa) + hiddenL @ Ta + ba, "sigmoid")
            r[:, i, :], dr[:, i, :] = self.activition(np.dot(dataT , Wb) + np.dot(hiddenL, Tb) + bb, "sigmoid")
            mul=np.dot((r[:, i, :] * hiddenL) , Tc)
            htil[:, i, :], dhtil[:, i, :] = self.activition(np.dot(dataT, Wc) + mul + bc, "tanh")
            h[:, i, :] = (1 - z[:, i, :]) * hiddenL + z[:, i, :] * htil[:, i, :]

            hiddenL = h[:, i, :]

        return hiddenL,{'z':z,'dz':dz,'r':r,'dr':dr,'h':h,'htil':htil,'dhtil':dhtil}

    def backGRU(self,data,cache,dEu):
        _, Ta, _, _, Tb, _, _, Tc, _=self.We.values()
        z=cache['z']
        dz=cache['dz']
        r=cache['r']
        dr=cache['dr']
        h=cache['h']
        htil=cache['htil']
        dhtil=cache['dhtil'] 
        dWa,dTa,dba,dWb,dTb,dbb,dWc,dTc,dbc=np.zeros((1,9))[0]
        numN=self.Size[1] #neuron number
        dims=np.shape(data)
        numS=dims[0] #sample number
        t=dims[1]    #time sequence
        sensor=dims[2]  #three sensor dim 

        for i in reversed(range(t)):
            dataT = data[:, i, :]

            if i > 0:
                hiddenL = h[:, i - 1, :]
            else:
                hiddenL = np.zeros((numS, numN))

            d_z = dEu * (htil[:, i, :] - hiddenL) * dz[:, i, :]
            dh_til = dEu * z[:, i, :] * dhtil[:, i, :]
            d_r = (np.dot(dh_til, Tc.T)) * hiddenL * dr[:, i, :]

            dWa += np.dot(dataT.T, d_z)
            dTa += np.dot(hiddenL.T , d_z)
            dba += np.sum(d_z,axis=0, keepdims=True)

            dWb += np.dot(dataT.T, d_r)
            dTb += np.dot(hiddenL.T , d_r)
            dbb += np.sum(d_r,axis=0, keepdims=True)

            dWc += np.dot(dataT.T , dh_til)
            dTc += np.dot(hiddenL.T, dh_til)
            dbc += np.sum(dh_til,axis=0, keepdims=True)

            d=0
            d +=  (1 - z[:, i, :])*dEu
            d += np.dot(d_z , Ta.T)
            d += np.dot(dh_til  , Tc.T) * (r[:, i, :] + hiddenL * (np.dot(dr[:, i, :] ,Tb.T)))

        return {"dWa": dWa, "dTa": dTa, "dba": dba,
                "dWb": dWb, "dTb": dTb, "dbb": dbb,
                "dWc": dWc, "dTc": dTc, "dbc": dbc}

    def learn(self,lr,mom,gFL,gMuLP):
        We=self.We
        inLayer=self.inLayer

        MLPparam=self.MLPparam
        MLPm=self.MLPm

        MLPlayer_len=self.MLPlayer_len

        for e in We:
            gUpd = "d"+ e
            inLayer[e] = lr * gFL[gUpd] + mom * inLayer[e]
            We[e] -= inLayer[e]

        for i in range(MLPlayer_len):
            MLPm["W"][i] =  lr * (gMuLP["W"][i]+mom * MLPm["W"][i])
            MLPm["b"][i] = lr * (gMuLP["b"][i] +mom * MLPm["b"][i]) 
            MLPparam["W"][i] -= MLPm["W"][i]
            MLPparam["b"][i] -= MLPm["b"][i]
            self.w_b_dict = We
            self.input_layer_m = inLayer
            self.mlp_w_b = MLPparam

        self.We=We
        self.inLayer=inLayer
        self.MLPparam=MLPparam
        self.MLPm=MLPm

    def train(self,data,gt,lr,mom,batchS,epoch,modelType):
        np.random.seed(555)
        lostTrain = []
        lostVal = []
        accTrain = []
        accVal = []

        sampleNum=data.shape[0]
        validation= sampleNum//10
        randomData = np.random.permutation(sampleNum)

        valData=data[randomData][:validation]
        valGt=gt[randomData][:validation]
        trainData=data[randomData][validation:]
        trainGt=gt[randomData][validation:]

        iteration=int(sampleNum/batchS)

        for i in range(epoch):
            s=0
            e=batchS
            rand = np.random.permutation(trainData.shape[0])
            trainData = trainData[rand]
            trainGt = trainGt[rand]
            for j in range(iteration):

                dataBatch = trainData[s:e]
                gtBatch = trainGt[s:e]

                choice,out,d,hidden,dHidden,cache = self.forward(dataBatch,modelType=modelType)
                dE = choice
                dE[gtBatch == 1] -= 1
                dE /= batchS

                gRNN, gMLP = self.backward(dataBatch, out, d, dE, modelType,hidden, dHidden, cache )
                self.learn(lr, mom, gRNN, gMLP)
                s = e
                e=e+ batchS

            trainacc, trainP, trainconf = self.choice(trainData,trainGt)
            trainloss = self.CE(trainGt, trainP)

            valAcc, valP, valConf = self.choice(valData,valGt)
            valloss = self.CE(valGt, valP)
            
            lostTrain.append(trainloss)
            lostVal.append(valloss)
            accTrain.append(trainacc)
            accVal.append(valAcc)
            print('\nTraining Loss: ', trainloss, 'Validation Loss: ', valloss, 'Training Accuracy: ', 'Validation Accuracy: ',trainacc, 'Epoch: ',i+1, 'out of', epoch)

            if (i > 15) and ( modelType!='gru'):
                avlost = lostVal[-9:-1] #check last 8 arbitrary choice
                avlost = sum(avlost) / len(avlost)

                limit = 0.005
                if (avlost - limit) < valloss < (avlost + limit):
                    print("\nTraining ends due to convergence.")
                    return {"lostTrain": lostTrain, "lostVal": lostVal,
                            "accTrain": accTrain, "accVal": accVal,'trainconf':trainconf}
        return {"lostTrain": lostTrain, "lostVal": lostVal,
                            "accTrain": accTrain, "accVal": accVal, 'trainconf':trainconf}

    def choice(self,data,gt=None):
        modelType=self.modelType0
        p,_,_,_,_,_=self.forward(data,modelType=modelType)
        predictions=p.argmax(axis=1)
        gt=gt.argmax(axis=1)

        accuracy=0
        confidence=np.zeros((6,6))
        for i in range(data.shape[0]):
            confidence[gt[i]][predictions[i]]+=1
            if gt[i]==predictions[i]:
                accuracy+=1
        accuracy/=len(data)
        return accuracy, p, confidence

    def CE(self,gt,choi):
        return np.sum(-gt*np.log(choi))/gt.shape[0]

    def activition(self,data,activation):
        if activation=='softmax':
            Z=np.exp(data)/np.sum(np.exp(data),axis=1,keepdims=True)
            d=None
        elif activation=='sigmoid':
            Z=1/(1+np.exp(-data))
            d=Z-Z**2
        elif activation=='relu':
            #Z=np.max(0.0,data)
            Z = data * (data > 0)
            d=1*(data>0)
        elif activation=='tanh':
            Z=np.tanh(data)
            d=1- Z**2
        return Z,d
#############End##################
import sys

question =sys.argv[1]
def sereftaha_kiremitci_21903711_miniproject(question):
    print('Question ',question)
    if question == '1' :
        Q1()
    elif question == '2' :
        q2()
    elif question == '3' :
        q3()

sereftaha_kiremitci_21903711_miniproject(question)

