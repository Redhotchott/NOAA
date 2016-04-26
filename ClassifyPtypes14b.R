#### SVM for the  ###

######ABOUT THIS SCRIPT#####
# This classify Ptypes script  is an 
# attempt to use SVM with Rbf  to 
# classify the data. Split by training set
# 0.6266077 accuracy. 
# Also has the accuracy function
# Only temp as input
# Outlyer Snow in miami has been removed. 

rm(list=ls())

library( 'e1071' )
library('rpart')
library('dplyr')
library('parallel')
library(sn)
library(fields)
library(mvtnorm)
library(foreach)
library(doSNOW)
load('Predictors.Rdata')

create.wt<-function(train.rows.mon){
  rain.rows=which(ptype[train.rows.mon]=="RA")
  snow.rows=which(ptype[train.rows.mon]=="SN")
  pellet.rows=which(ptype[train.rows.mon]=="IP")
  ice.rows=which(ptype[train.rows.mon]=="FZRA")
  
  r.l<-length(rain.rows)
  s.l<-length(snow.rows)
  p.l<-length(pellet.rows)
  i.l<-length(ice.rows)
  p.lengths<-c(i.l,p.l,r.l,s.l)
  p.class<-p.lengths!=0
  
  ref<-which(p.lengths==min(p.lengths[p.lengths!=0]))
  class.wts<-p.lengths[ref]/p.lengths[p.class]
  return(class.wts)
}

conf.mat.create.ipfzra<-function(tt, test.rows.mon,zz.set){
  if(dim(tt)[1]==2 & dim(tt)[2]==2){
    zz.set<-zz.set+tt
  } else if (dim(tt)[1]==2 &dim(tt)[2]==1){
    if(any(colnames(zz.set)=='1')){
      zz.set[1:2,1]<-zz.set[1:2,1]+tt
    } else {
      zz.set[1:2,2]<-zz.set[1:2,2]+tt
    }
  } else if (dim(tt)[1]==1&dim(tt)[2]==2){
    if(any(rownames(zz.set)=='1')){
      zz.set[1,1:2]<-zz.set[1,1:2]+tt
    } else {
      zz.set[2,1:2]<-zz.set[2,1:2]+tt
    }
  } 
  return(zz.set)
}

accuracy<-function(zz){ 
  tot<-sum(zz)
  diags<-function(zz1){return(sum(diag(zz1)))}
  right<-sum(apply(zz,3,diags))
  return(right/tot)
}

ptype.fac<-as.factor(ptype)

Twb.type<-cbind(Twb.prof,ptype.fac) %>% as.data.frame
colnames(Twb.type)<-c("H0","H1","H2", "H3","H4","H5","H6","H7", "H8","H9","H10","H11","H12","H13","H14","H15","H16","H17","H18","H19","H20","H21","H22","H23","H24","H25","H26","H27","H28","H29","H30","ptype.df")
attach(Twb.type)


years=as.numeric(substr(dates,1,4))
months=as.numeric(substr(dates,5,6))
all.months=as.numeric(substr(dates[date.ind],5,6))

test.nn=array()
train.nn=array()
truth<- array()
pred<- array()
zz<-array(NA, c(2,2,12))
model<-list()
model.mon<-list()
res<-list()
res.mon<-list()

for ( i in 1:12){
  train.years=1996:2000+i-1
  test.years=2000+i
  
  print(paste('Training Set: ', i))
  
  train.labels=head(which((years>=train.years[1] & months >8)),1):tail(which(years<=train.years[5]+1 & months <6),1)
  test.labels=which((years==test.years & months>8) | (years==test.years+1 & months < 6))
  
  
  train.rows=which(date.ind%in%train.labels)
  test.rows=which(date.ind%in%test.labels)
  
  train.rows.ip=train.rows[which(ptype[train.rows]=='IP')]
  train.rows.fzra=train.rows[which(ptype[train.rows]=='FZRA')]
  train.rows=c(train.rows.ip,train.rows.fzra)
  
  test.rows.ip=test.rows[which(ptype[test.rows]=='IP')]
  test.rows.fzra=test.rows[which(ptype[test.rows]=='FZRA')]
  test.rows=c(test.rows.ip,test.rows.fzra)
  
  train.nn[i]=length(train.rows)
  test.nn[i]=length(test.rows)
  
  t.w<-create.wt(train.rows)
  model.mon[[i]]<-svm( ptype.df~., data=Twb.type[train.rows,],
                       probability=T, type='C-classification', 
                       class.weights=c("1"=t.w[1], "2"=t.w[2]))
  
  res.mon[[i]]<-predict(model.mon[[i]], newdata=Twb.type[test.rows,1:31],decision.values=T)
    
  zz[,,i]<-table(pred = res.mon[[i]], true = Twb.type[test.rows,32])
  
  print(zz[,,i])
}



