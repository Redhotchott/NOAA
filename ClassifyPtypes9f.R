######ABOUT THIS SCRIPT#####
# This classify Ptypes script will attempt 
# to plot the loss function versus iteration 
# as well as the accurracy on the CV set. 
# We also want to balance the training classes. 
# Only temp as input
# THIS IS AN ATTEMPT AT SIGMOID ACTIVATION AND LOGISTIC cost function
# Using format of the RSNNS package - better than 9e
# Outlyer Snow in miami has been removed. 

rm(list=ls())
setwd('/Users/tchott/Documents/Capstone')
load("predictors.RData")
library(ggplot2)
library(caret)
library(plot3D)
library(RSNNS)

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
par(mfrow=c(1,1))
z <- sample(1:nrow(Twb.prof),1)
displayVTempR(z)

#convert to data frame 
df<-as.data.frame(Twb.prof)
#labeling the columns
colnames(df)<-c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)
#formating the ptypes into factors
f.ptype<-as.factor(ptype)
#create one df to make it easier 
noaa<-cbind(df,f.ptype)

##Creating Testing, CV, and Training Set
ptypeValues <- noaa[,1:31]
ptypeTargets <-decodeClassLabels(noaa[,32], valTrue=.91,valFalse=0.03)
#irisTargets <- decodeClassLabels(iris[,5], valTrue=0.9, valFalse=0.1)
# should i keep them as 100% class values. not sure why in the iris example they 
# made the factors not 100% either they are or they aren't

# need an alternate way to split these up, as we want balanced testing classes, 
# so I can't do it like the tutorial, shown below. However need it to a list of dim4,
# inputsTrain, targetsTrain, inputsTest, targetsTest as matrices
# iris <- splitForTrainingAndTest(irisValues, irisTargets, ratio=0.15)
full<-seq(1,length(ptype))
set.seed(3456)
inTrain<-c(sample(which(ptype=='RA'),750), sample(which(ptype=='SN'),750),sample(which(ptype=='FZRA'),750),sample(which(ptype=='IP'),750) )

inputsTrain<-as.matrix(ptypeValues[inTrain,1:31])
inputsTest<-as.matrix(ptypeValues[-inTrain,1:31])
targetsTrain<-as.matrix(ptypeTargets[inTrain,])
targetsTest<-as.matrix(ptypeTargets[-inTrain,])

#iris <- normTrainingAndTestSet(iris)  #can't do this way, bc currently my data not formatted the same
inputsTrain<-normalizeData(inputsTrain)
inputsTest<-normalizeData(inputsTest)
# get normalizing values by getNormParameters

# eventually want to do it the way below
# precip <-normTrainingAndTestSet(precip)

precip<-list()
precip[[1]]<-inputsTrain
precip[[2]]<-targetsTrain
precip[[3]]<-inputsTest
precip[[4]]<-targetsTest

names(precip)[1]<-"inputsTrain" 
names(precip)[2]<-"targetsTrain" 
names(precip)[3]<-"inputsTest" 
names(precip)[4]<-"targetsTest" 


#model <- mlp(iris$inputsTrain, iris$targetsTrain, size=5, learnFuncParams=c(0.1), maxit=50, inputsTest=iris$inputsTest, targetsTest=iris$targetsTest)
model4 <- mlp(inputsTrain, targetsTrain, size=4, learnFunc = "Std_Backpropagation", learnFuncParams=c(0.1),
             maxit=100, inputsTest=inputsTest, targetsTest=targetsTest)
#learning function parameters are the learning rate, can also have the maximum output difference for the min before it considers it no differ.
model10 <- mlp(inputsTrain, targetsTrain, size=10, learnFunc = "Std_Backpropagation", learnFuncParams=c(0.1),
              maxit=100, inputsTest=inputsTest, targetsTest=targetsTest)


saveRDS(model4, 'model9f4h.RDS')
summary(model4)
weightMatrix(model4)
extractNetInfo(model4)

saveRDS(model10, 'model9f10h.RDS')

par(mfrow=c(2,2))
#plotIterativeError(model4)

predictions4 <- predict(model4,precip$inputsTest)
predictions10 <-predict(model10,precip$inputsTest)

#plotRegressionError(predictions4[,2], precip$targetsTest[,2])

confusionMatrix(precip$targetsTrain,fitted.values(model4))
conf.mat4<-confusionMatrix(precip$targetsTest,predictions4)
conf.mat10 <-confusionMatrix(precip$targetsTest,predictions10)

#plotROC(fitted.values(model4)[,2], precip$targetsTrain[,2])
#title("Trained Data")
#plotROC(predictions4[,2], precip$targetsTest[,2])
#title("Test Data")
#receiver operating characteristic curve 
#false positive rate versus true positive rate (think confusion matrix)

#confusion matrix with 402040-method
conf.matTrain<-confusionMatrix(precip$targetsTrain, encodeClassLabels(fitted.values(model), method="402040", l=0.4, h=0.6))
saveRDS(conf.mat4, 'confmat9f4h.RDS')
saveRDS(conf.mat10, 'confmat9f10h.RDS')
conf.mat4<-readRDS('confmat9f4h.RDS')
conf.mat10<-readRDS('confmat9f4h.RDS')
conf.mat10
conf.mat4 

#More code and plots for this package: http://dicits.ugr.es/software/RSNNS/?view=Examples%20of%20high-level%20API#i2
par(mfrow=c(1,1))
hist(model$fitted.values - precip$targetsTrain)
#don't use above, only for linear models

#THIS IS THE MODEL COMPARISON!!! WHOOOOOO
parameterGrid <-  expand.grid(c(4,6,8,10), c(0.00316, 0.0147, 0.1)) 
colnames(parameterGrid) <-c("nHidden", "learnRate") 
rownames(parameterGrid) <-paste("nnet-", apply(parameterGrid, 1, function(x) {paste(x,sep="", collapse="-")}), sep="") 
models<-apply(parameterGrid, 1, function(p) { 
  mlp(precip$inputsTrain, precip$targetsTrain, size=p[1], learnFunc="Std_Backpropagation", 
  learnFuncParams=c(p[2], 0.1), maxit=200, inputsTest=precip$inputsTest, 
  targetsTest=precip$targetsTest) 
  })

#Shows the iterative error on each model
par(mfrow=c(4,3)) 
for(modInd in 1:length(models)) { 
  plotIterativeError(models[[modInd]], main=names(models)[modInd]) 
}

#now look at training and testing RMSE's and see which preform best
trainErrors <-  data.frame(lapply(models, function(mod) { 
  error<-sqrt(sum((mod$fitted.values - precip$targetsTrain)^2)) 
  error 
})) 
testErrors <-  data.frame(lapply(models, function(mod) { 
  pred <-  predict(mod,precip$inputsTest) 
  error <-  sqrt(sum((pred - precip$targetsTest)^2)) 
  error 
})) 
t(trainErrors)
t(testErrors)

#choosing your best model: 
trainErrors[which(min(trainErrors) == trainErrors)]
testErrors[which(min(testErrors) == testErrors)]
model<-models[[which(min(testErrors) == testErrors)]]


#maps
model<-som(irisValues, mapX = 16, mapY = 16, maxit = 500, targets = ptypeTargets)


#som creates and trains a self organizing map with one layer 
plotActMap(model$map, col = rev(heat.colors(12))) 


plotActMap(log(model$map + 1), col = rev(heat.colors(12))) 
persp(1:model$archParams$mapX, 1:model$archParams$mapY, log(model$map + 1), theta = 30, phi = 30, expand = 0.5, col = "lightblue") 
plotActMap(model$labeledMap)

for(i in 1:ncol(ptypeValues)) plotActMap(model$componentMaps[[i]], col = rev(topo.colors(12)))

