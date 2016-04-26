#### Set up NN on data ###

######ABOUT THIS SCRIPT#####
# Plotting the IP and FZRA observations within 
# each training set, to see if, with the naked eye
# I can discriminate between the two different types
# of observations. 
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

nptype<-matrix(0,nrow=nrow(Twb.prof),ncol=1)  ##Setting up my pytpes as a 4 columns indicating ptype
for ( i in 1:nrow(Twb.prof)){
  if (ptype[i]=='RA') {nptype[i]<-1}
  if (ptype[i]=='SN') {nptype[i]<-2}
  if (ptype[i]=='FZRA') {nptype[i]<-3}
  if (ptype[i]=='IP') {nptype[i]<-4}
}

nnet.mnist.prev<-list()
par(mfrow=c(1,2))
nnet <- function(X, Y, Xcv.proc, ycv, step_size = 0.5, reg = 0.05, h = 10, niteration){  
  # get dim of input  
  N <- nrow(X) # number of examples  
  K <- ncol(Y) # number of classes  
  D <- ncol(X) # dimensionality   
  # initialize parameters randomly  
  set.seed(3456)
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  
  b <- matrix(0, nrow = 1, ncol = h)  
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)  
  b2 <- matrix(0, nrow = 1, ncol = K) 
  # gradient descent loop to update weight and bias
  lossvec<-rep(0,niteration)
  cvaccvec<-rep(0,niteration)
  loss.prev<-100
  acc.prev<-.01
  acc.now<-0.011
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
    if(i>300){
      if(loss>loss.prev | acc.prev>acc.now){
        print(paste('Stopping at iteration: ', i))
        break
      }
    }
    
    loss.prev<-loss
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
    nnet.mnist.prev<-nnet.mnist
    nnet.mnist<-list(W, b, W2, b2)
    if(i>300 & loss>loss.prev){print('reached')}
    predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
    acc.now<-mean(predicted_class == (ycv))
    cvaccvec[i] <-mean(predicted_class == (ycv))
    acc.prev<-cvaccvec[i-1]
    if (i%%100 == 0 | i == niteration){      
      print(paste("iteration", i,': accuracy', mean(predicted_class==(ycv))))
    }  
  }  
  plot(lossvec, main='Learning Curve', xlab='iteration', ylab='loss', type='n')
  lines(lossvec)
  plot(cvaccvec, main='Accuracy', xlab='iteration', ylab='accuracy', type='n')
  lines(cvaccvec)
  return(list(nnet.mnist.prev,lossvec,cvaccvec,predicted_class))
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




train.nn<-array()
test.nn<-array()
bal.train.rows<-list()
bal.test.rows<-list()
bal.cv.rows<-list()
reference.rows<-array()
ip.length<-array()
fz.length<-array()

##Creating Testing, CV, and Training Sets
par(mfrow=c(1,1))
for(i in 1:12){
  train.years=1996:2000+i-1
  test.years=2000+i
  
  print(i)
  
  train.labels=head(which((years>=train.years[1] & months >8)),1):tail(which(years<=train.years[5]+1 & months <6),1)
  test.labels=which((years==test.years & months>8) | (years==test.years+1 & months < 6))
  
  set.seed(3456)
  train.rows=which(date.ind%in%train.labels)
  test.rows=which(date.ind%in%test.labels)
  ptype.temp<-ptype[train.rows]
  ip.length[i]<-length(which(ptype.temp=='IP'))
  fz.length[i]<-length(which(ptype.temp=='FZRA'))
  ip.rows<-which(ptype.temp=='IP')
  fz.rows<-which(ptype.temp=='FZRA')
  ip.mean<-apply(Twb.prof[train.rows[ip.rows],], 2, mean)
  fz.mean<-apply(Twb.prof[train.rows[fz.rows],], 2, mean)
  step.size=seq(0,3000,by=100)
  plot(ip.mean,step.size,xlab="Temperature (K)", col='red',xlim=range(Twb.prof[,]),ylab="Meters AGL",type="l",main=paste("Mean of IP vs FZRA Set: ", i))
  abline(v=273.15, col='black')
  lines(fz.mean, step.size, col='blue')
  legend('topleft', c('FZRA', 'IP'), col=c('blue', 'red'), pch=19)
  train.bal<-c(sample(which(ptype.temp=='IP'),25),sample(which(ptype.temp=='FZRA'),25) )
  bal.train.rows[[i]]<-train.rows[train.bal]
  bal.cv.rows[[i]]<-train.rows[-train.bal]
  bal.test.rows[[i]]<-test.rows
  train.nn[i]=length(train.rows)
  test.nn[i]=length(test.rows)
  reference.rows<-c(reference.rows,test.rows)
}
reference.rows<-reference.rows[2:length(reference.rows)]

## Plotting the IP and FZRA obs from each training set
step.size=seq(0,3000,by=100)
par(mfrow=c(1,2))
for(i in 1:12){
  ip.rows<-bal.train.rows[[i]][1:25]
  fz.rows<-bal.train.rows[[i]][(26):length(bal.train.rows[[i]])]
  step.size.mat<-matrix(step.size,nrow=length(fz.rows), ncol=length(step.size), byrow=T)
  matplot(t(Twb.prof[fz.rows,]),t(step.size.mat),type='l', col='blue', xlim=c(240,290), xlab='Temperature', ylab='Meters AGL', main=paste('IP vs FZRA Obs Set: ', i))
  step.size.mat<-matrix(step.size,nrow=length(ip.rows), ncol=length(step.size), byrow=T)
  matlines(t(Twb.prof[ip.rows,]),t(step.size.mat),type='l', col='red')
  abline(v=273.15, col='black')
  legend('topleft', c('FZRA', 'IP'), col=c('blue', 'red'), pch=19)
}




