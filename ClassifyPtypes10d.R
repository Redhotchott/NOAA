#### Set up NN on data ###

######ABOUT THIS SCRIPT#####
# This classify Ptypes script will attempt 
# to balance the training classes across 12 
# traininng sets 12 disjoint testing sets.
# Overall 90.67% accuracy 
# with 1000 iterations with the stopping procedure. 
# This script modifys the 10b to more easily access
# and analyze the misclassified observations. 
# Only temp as input
# Outlyer Snow in miami has been removed. 

rm(list=ls())
setwd('/Users/tchott/Documents/Capstone')
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

##Creating Testing, CV, and Training Sets
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
  train.bal<-c(sample(which(ptype.temp=='RA'),150), sample(which(ptype.temp=='SN'),150),sample(which(ptype.temp=='FZRA'),150),sample(which(ptype.temp=='IP'),150) )
  bal.train.rows[[i]]<-train.rows[train.bal]
  bal.cv.rows[[i]]<-train.rows[-train.bal]
  bal.test.rows[[i]]<-test.rows
  train.nn[i]=length(train.rows)
  test.nn[i]=length(test.rows)
  reference.rows<-c(reference.rows,test.rows)
}
reference.rows<-reference.rows[2:length(reference.rows)]

## Creating the 12 neural networks
noaa.nnet<-list()
nnet.mnist<-list()
predicted<-array()
true<-array()
for (i in 1:12){
  print(paste('SET NUMBER: ', i))
  X <- Twb.prof[bal.train.rows[[i]],] #data matrix (each row = single example)
  N <- nrow(X)# number of examples 
  y <- nptype[bal.train.rows[[i]],] # class labels
  K <- length(unique(y)) #number of classes
  X.proc <- X[,]/max(X) #scale (temp data)
  D <- ncol(X.proc) #dimensionality 
  Xcv <- Twb.prof[bal.cv.rows[[i]],] #testing data
  ycv <-nptype[bal.cv.rows[[i]],]# class labels
  Xcv.proc <- Xcv[,]/max(X) # scale CV data
  Xt <- Twb.prof[bal.test.rows[[i]],] #testing data
  yt <-nptype[bal.test.rows[[i]],]# class labels
  Xt.proc <- Xt[,]/max(X) # scale CV data
  Y <-matrix(0, N, K) 
  for (j in 1:N){
    Y[j, y[j]]<- 1
  }
  set.nnet <- nnet(X.proc, Y, Xcv.proc, ycv, h=10, step_size = 0.3, reg = 0.00, niteration =1000)
  noaa.nnet[[i]]<-set.nnet
  predicted_class <- nnetPred(Xcv.proc, set.nnet[[1]])
  print(paste('cv set accuracy:', mean(predicted_class == (ycv))))
  predicted_class <- nnetPred(Xt.proc, set.nnet[[1]])
  print(paste('testing accuracy:',mean(predicted_class == (yt))))
  predicted<-c(predicted,predicted_class)
  true<-c(true,nptype[bal.test.rows[[i]]])
}
predicted<-predicted[2:length(predicted)]
true<-true[2:length(true)]
saveRDS(noaa.nnet, "noaannet10d.rds")
saveRDS(predicted, "predicted10d.rds")
saveRDS(true, "true10d.rds")
print(paste('overall testing accuracy: ', mean(predicted==true)))

confusionMatrix(predicted,true)

#lets choose a random SNOW, IP and FZ Rain Observation and plot

par(mfrow=c(1,3))
ra<-sample(which(nptype==1), size=1)
sn<-sample(which(nptype==2), size=1)
fz<-sample(which(nptype==3), size=1)
ip<-sample(which(nptype==4), size=1)
#displayVTempR(ra)
displayVTempR(sn)
displayVTempR(fz)
displayVTempR(ip)

#Now for the misclassified from nnet: 
par(mfrow=c(1,1))
rat.fzp<-sample(which(predicted==3&true==1),size=1)
displayVTempR(reference.rows[rat.fzp])

fzt.rap<-sample(which(predicted==1&true==3),size=1)
displayVTempR(reference.rows[fzt.rap])



