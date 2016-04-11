#### Set up NN on data ###

######ABOUT THIS SCRIPT#####
# This classify Ptypes script will attempt 
# to just differentiate between IP and FZRA.  
# Balanced training sets 
# current issue- depends on initialization of 
# weights. If they are lucky, we reach about 85% 
# and it shows, but usually means that most are classified as 
# FZRA. Very easy to fall into a FZRA dominant hole. 
# Only temp as input
# Outlyer Snow in miami has been removed. 

rm(list=ls())
setwd('/Users/tchott/Documents/NOAA')
load("predictors.RData")
library(ggplot2)
library(caret)
library(plot3D)
library(e1071)

cols=1:32  	#Columns (i.e. levels) of the temperature profiles to be used in the 
years=as.numeric(substr(dates,1,4))
months=as.numeric(substr(dates,5,6))
all.months=as.numeric(substr(dates[date.ind],5,6))

Twb.prof<-Twb.prof[-41977,]
date.ind<-date.ind[-41977]
ptype<-ptype[-41977]
station.ind<-station.ind[-41977]

displayVTempR <- function(X){
  step.size=seq(0,3000,by=100)
  plot(Twb.prof[X,],step.size,xlab="Temperature (K)", xlim=range(Twb.prof[,]),ylab="Meters AGL",type="l",main=paste("Random Observation of Type: ", ptype[X]))
  abline(v=273.15, col='Red')
  print(paste("Precipitation Type:", ptype[X]))
}

z <- sample(1:nrow(Twb.prof),1)
displayVTempR(z)

ip.fz.rows<-which(ptype=='IP'|ptype=='FZRA')
ptype.red<-ptype[ip.fz.rows]
Twb.red<-Twb.prof[ip.fz.rows,]

nptype<-matrix(0,nrow=nrow(Twb.red),ncol=1)  ##Setting up my pytpes as a 2 columns indicating ptype
for ( i in 1:nrow(Twb.red)){
  if (ptype.red[i]=='FZRA') {nptype[i]<-1} 
  if (ptype.red[i]=='IP') {nptype[i]<-2}
}

#752 IP, so we will split 300,100,352 in train, cv, and test of IP. 
##Creating Testing, CV, and Training Set
full<-seq(1,length(ptype.red))
set.seed(3456)
inTrain<-c(sample(which(ptype.red=='IP'),300), sample(which(ptype.red=='FZRA'),300))
training <- Twb.red[inTrain, ]
full<-full[-inTrain]
inCV<-createDataPartition(y=full, p=0.5,list=F)
CV<-Twb.red[inCV,]
inTest<-full[-inCV]
testing<-Twb.red[inTest,]



X <- training #data matrix (each row = single example)
N <- nrow(X)# number of examples 
y <- nptype[inTrain,] # class labels
K <- length(unique(y)) #number of classes
X.proc <- X[,]/max(X) #scale (temp data)
D <- ncol(X.proc) #dimensionality 
Xcv <- CV #testing data
ycv <-nptype[inCV,]# class labels
Xcv.proc <- Xcv[,]/max(X) # scale CV data
Xt <- testing #testing data
yt <-nptype[inTest,]# class labels
Xt.proc <- Xt[,]/max(X) # scale CV data
Y <-matrix(0, N, K) 
for (i in 1:N){
  Y[i, y[i]]<- 1
}

par(mfrow=c(1,2))
nnet <- function(X, Y, Xcv.proc, ycv, step_size = 0.5, reg = 0.05, h = 10, niteration){  
  # get dim of input  
  N <- nrow(X) # number of examples  
  K <- ncol(Y) # number of classes  
  D <- ncol(X) # dimensionality   
  # initialize parameters randomly  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  
  b <- matrix(0, nrow = 1, ncol = h)  
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)  
  b2 <- matrix(0, nrow = 1, ncol = K)   
  # gradient descent loop to update weight and bias
  lossvec<-rep(0,niteration)
  cvaccvec<-rep(0,niteration)
  for (i in 0:niteration){    
    # hidden layer, ReLU activation    
    hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))    
    hidden_layer <- matrix(hidden_layer, nrow = N)    
    # class score    
    scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)  
    # compute and normalize class probabilities    
    exp_scores <- exp(scores)    
    probs <- exp_scores / rowSums(exp_scores)     
    # compute the loss: sofmax and regularization    
    correct_logprobs <- -log(probs)
    data_loss <- sum(correct_logprobs*Y)/N    
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2)    
    loss <- data_loss + reg_loss  
    lossvec[i]<-loss
    # check progress    
    if (i%%100 == 0 | i == niteration){      
      print(paste("iteration", i,': loss', loss))
    }     
    # compute the gradient on scores    
    dscores <- probs-Y    
    dscores <- dscores/N     
    # backpropate the gradient to the parameters    
    dW2 <- t(hidden_layer)%*%dscores    
    db2 <- colSums(dscores)   
    # next backprop into hidden layer    
    dhidden <- dscores%*%t(W2)    
    # backprop the ReLU non-linearity    
    dhidden[hidden_layer <= 0] <- 0    
    # finally into W,b    
    dW <- t(X)%*%dhidden    
    db <- colSums(dhidden)     
    # add regularization gradient contribution    
    dW2 <- dW2 + reg *W2    
    dW <- dW + reg *W     
    # update parameter     
    W <- W-step_size*dW    
    b <- b-step_size*db    
    W2 <- W2-step_size*dW2    
    b2 <- b2-step_size*db2  
    nnet.mnist<-list(W, b, W2, b2)
    predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
    cvaccvec[i] <-mean(predicted_class == (ycv))
    if (i%%100 == 0 | i == niteration){      
      print(paste("iteration", i,': accuracy', mean(predicted_class==(ycv))))
    }  
  }  
  plot(lossvec, main='Learning Curve', xlab='iteration', ylab='loss', type='n')
  lines(lossvec)
  plot(cvaccvec, main='Accuracy', xlab='iteration', ylab='accuracy', type='n')
  lines(cvaccvec)
  return(list(W, b, W2, b2,lossvec,cvaccvec,predicted_class))
}

nnetPred<-function(X,para=list()){
  W<-para[[1]]
  b<-para[[2]] 
  W2<-para[[3]]
  b2<-para[[4]]
  N<-nrow(X)
  hidden_layer <- pmax(0, X%*%W + matrix(rep(b,N), nrow = N, byrow = T))
  hidden_layer <- matrix(hidden_layer, nrow = N)
  scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)
  predicted_class<-apply(scores,1,which.max)
  return(predicted_class)
}

nnet.mnist <- nnet(X.proc, Y, Xcv.proc, ycv, h=10, step_size = 0.3, reg = 0.001, niteration =150)
predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
print(paste('cv set accuracy:', mean(predicted_class == (ycv))))

predicted_class <- nnetPred(Xt.proc, nnet.mnist)
print(paste('testing accuracy:',mean(predicted_class == (yt))))

confusionMatrix(predicted_class,yt)

