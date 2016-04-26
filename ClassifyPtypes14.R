#### Set up NN on data ###

######ABOUT THIS SCRIPT#####
# This classify Ptypes script  is an 
# attempt to use SVM with Rbf  to 
# classify the data. Split by month. 
# currently not working, one svm name in 1,5 is missing or misspelled. 
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
  
  #######################################################
  ##Computing means and covariances for each precip type
  #######################################################
  for (j in unique(all.months)){
    print(paste('Month: ', j))
    train.rows.mon=train.rows[which(all.months[train.rows]==j)]
    test.rows.mon=test.rows[which(all.months[test.rows]==j)]
    
    
    
    #in order F, I, R, S
    t.w<-create.wt(train.rows.mon)
    
    if(length(t.w)==3){
      if(any(ptype[train.rows.mon]=='IP')){model.mon[[j]]<- svm( ptype.df~., data=Twb.type[train.rows.mon,],
                                                                 probability=T, type='C-classification', 
                                                                 class.weights=c("2"=t.w[1], "3"=t.w[2], "4"=t.w[3]))
      } else {model.mon[[j]]<- svm( ptype.df~., data=Twb.type[train.rows.mon,],
                                    probability=T, type='C-classification', 
                                    class.weights=c("1"=t.w[1], "3"=t.w[2], "4"=t.w[3]))}
    } else if (length(t.w)==4) {model.mon[[j]]<- svm( ptype.df~., data=Twb.type[train.rows.mon,],
                                  probability=T, type='C-classification', 
                                  class.weights=c("1"=t.w[1], "2"=t.w[2], "3"=t.w[3],"4"=t.w[4]))
    } else if (length(t.w)==2){ model.mon[[j]]<- svm( ptype.df~., data=Twb.type[train.rows.mon,],
                            probability=T, type='C-classification', 
                            class.weights=c("1"=t.w[1], "2"=t.w[2]))
    } 
    if(length(unique(ptype[train.rows.mon])<2)){res.mon[[j]]}
    else {
      res.mon[[j]] <- predict( model.mon[[j]], newdata=Twb.type[test.rows.mon,1:31], decision.values = T)
      }
    tt<-table(pred = res.mon[[j]], true = Twb.type[test.rows.mon,32])
    if(j==11){zz[,,i]<-matrix(0,nrow=2,ncol=2)}
    zz[,,i]<-conf.mat.create.ipfzra(tt,test.rows.mon,zz[,,i])
    #     if(j==11){
    #       zz[,,i]<-tt
    #     } else if (dim(tt)[1]==3 & dim(tt)[2]==2){
    #       zz[c(1,3:4),3:4,i]<-zz[c(1,3:4),3:4,i]+tt
    #     } else if (dim(tt)[2]<4 & any(ptype[train.rows.mon]=='IP')){
    #       zz[2:4,2:4,i]<-zz[2:4,2:4,i]+tt
    #     } else if (dim(tt)[2]<4 & any(ptype[train.rows.mon]=='FZRA')){
    #       zz[c(1,3:4),c(1,3:4),i]<-zz[c(1,3:4),c(1,3:4),i]+tt
    #     } else {zz[,,i]<-zz[,,i]+tt}
    
    model[[i]]<-model.mon
    res[[i]]<-res.mon
    
    # print(tt)
    
    ##FUNCTIONIZE zz
  }
  print(zz[,,i])
}

