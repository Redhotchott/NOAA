#### Set up NN on data ###

######ABOUT THIS SCRIPT#####
# This classify Ptypes script will attempt 
# to just differentiate between IP and FZRA.  
# Balanced IP/FZRA through the train, CV, and test sets
# working on the break in the loop currently. 
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

##REDUCING TO JUST IP AND FZRA
ipfz.rows<-which(ptype=='FZRA' |ptype=="IP")
ip.rows<-which(ptype=='IP')
Twb.prof<-Twb.prof[ipfz.rows,]
ptype<-ptype[ipfz.rows]

nptype<-matrix(0,nrow=nrow(Twb.prof),ncol=1)  ##Setting up my pytpes as a 4 columns indicating ptype
for ( i in 1:nrow(Twb.prof)){
  if (ptype[i]=='FZRA') {nptype[i]<-1}
  if (ptype[i]=='IP') {nptype[i]<-2}
}

par(mfrow=c(1,2))
nnet <- function(X, Y, Xcv.proc, ycv, step_size = 0.5, reg = 0.05, h = 10, niteration){ 
  acc.prev=0
  # set up linear decay of the example: 
  decay<-seq(from=step_size,to=0, by=-step_size/(niteration-1))
  decay<-c(step_size,decay)
  # get dim of input  
  N <- nrow(X) # number of examples  
  K <- ncol(Y) # number of classes  
  D <- ncol(X) # dimensionality   
  # initialize parameters randomly  
  set.seed(101)
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
    W <- W-decay[i+1]*dW    
    b <- b-decay[i+1]*db    
    W2 <- W2-decay[i+1]*dW2    
    b2 <- b2-decay[i+1]*db2  
    nnet.prev<-nnet.mnist
    nnet.mnist<-list(W, b, W2, b2)
    predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
    cvaccvec[i] <-mean(predicted_class == (ycv))
    if(i>1){if(cvaccvec[i]<cvaccvec[i-1]){break}}
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


full<-1:length(ptype)
train.bal<-c(sample(which(ptype=='FZRA'),250),sample(which(ptype=='IP'),250) )
full<-full[-train.bal]
cv.bal<-c(sample(which(ptype[full]=='FZRA'),250),sample(which(ptype[full]=='IP'),250) )
full<-full[-cv.bal]
test.bal<-c(sample(which(ptype[full]=='FZRA'),250),sample(which(ptype[full]=='IP'),250))

X <- Twb.prof[train.bal,] #data matrix (each row = single example)
N <- nrow(X)# number of examples 
y <- nptype[train.bal,] # class labels
K <- length(unique(y)) #number of classes
X.proc <- X[,]/max(X) #scale (temp data)
D <- ncol(X.proc) #dimensionality 
Xcv <- Twb.prof[cv.bal,] #testing data
ycv <-nptype[cv.bal,]# class labels
Xcv.proc <- Xcv[,]/max(X) # scale CV data
Xt <- Twb.prof[test.bal,] #testing data
yt <-nptype[test.bal,]# class labels
Xt.proc <- Xt[,]/max(X) # scale CV data
Y <-matrix(0, N, K) 
for (j in 1:N){
  Y[j, y[j]]<- 1
}

nnet.mnist <- nnet(X.proc, Y, Xcv.proc, ycv, h=10, step_size = 0.1, reg = 0, niteration =1000)

predicted_class <- nnetPred(Xcv.proc, nmnist)
print(paste('cv set accuracy:', mean(predicted_class== (ycv))))
predicted_class <- nnetPred(Xt.proc, nmnist)
  print(paste('testing accuracy:',mean(predicted_class == (yt))))
  predicted<-c(predicted,predicted_class)
  true<-c(true,nptype[bal.test.rows[[i]]])
}
predicted<-predicted[2:length(predicted)]
true<-true[2:length(true)]
#saveRDS(noaa.nnet, "noaannet10e.rds")
#saveRDS(predicted, "predicted10e.rds")
#saveRDS(true, "true10e.rds")
print(paste('overall testing accuracy: ', mean(predicted==true)))

confusionMatrix(predicted,true)



train.nn<-array()
test.nn<-array()
bal.train.rows<-list()
bal.test.rows<-list()
bal.cv.rows<-list()
reference.rows<-array()
ip.length<-array()
fz.length<-array()

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
  ptype.temp<-ptype[test.rows]
  ip.length[i]<-length(which(ptype.temp=='IP'))
  fz.length[i]<-length(which(ptype.temp=='FZRA'))
  
}

sum(ip.length)
sum(fz.length)


