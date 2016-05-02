#####Slide Graphics####
rm(list=ls())

load('predictors.RData')

##OVERVIEW
par(mfrow=c(1,1))
step.size=seq(0,3000,by=100)
fz.rows<-which(ptype=='FZRA')
ip.rows<-which(ptype=='IP')
sn.rows<-which(ptype=='SN')
ra.rows<-which(ptype=='RA')
step.size.mat<-matrix(step.size,nrow=25, ncol=length(step.size), byrow=T)
matplot(t(Twb.prof[fz.rows[1:25],]),t(step.size.mat),type='l',lty='solid', col='blue', xlim=c(240,290), xlab='Temperature', ylab='Meters AGL', main=paste('Precipitation Type Overview'))
step.size.mat<-matrix(step.size,nrow=25, ncol=length(step.size), byrow=T)
matlines(t(Twb.prof[ip.rows[1:25],]),t(step.size.mat),type='l',lty='solid', col='red')
step.size.mat<-matrix(step.size,nrow=25, ncol=length(step.size), byrow=T)
matlines(t(Twb.prof[sn.rows[1:25],]),t(step.size.mat),type='l',lty='solid', col='orange')
step.size.mat<-matrix(step.size,nrow=25, ncol=length(step.size), byrow=T)
matlines(t(Twb.prof[ra.rows[1:25],]),t(step.size.mat),type='l', lty='solid', col='green')
abline(v=273.15, col='black')
legend('topleft', c('FZRA', 'IP', 'SN', 'RA'), col=c('blue', 'red', 'orange', 'green'), pch=19)


##IP VS FZRA
par(mfrow=c(1,1))
step.size=seq(0,3000,by=100)
ip.rows<-which(ptype=='IP')
fz.rows<-which(ptype=='FZRA')
step.size.mat<-matrix(step.size,nrow=50, ncol=length(step.size), byrow=T)
matplot(t(Twb.prof[fz.rows[51:100],]),t(step.size.mat),type='l', col='blue', lty='solid', xlim=c(240,290), xlab='Temperature', ylab='Meters AGL', main=paste('IP vs FZRA'))
step.size.mat<-matrix(step.size,nrow=50, ncol=length(step.size), byrow=T)
matlines(t(Twb.prof[ip.rows[51:100],]),t(step.size.mat),type='l', lty='dashed', col='red')
abline(v=273.15, col='black')



ip.mean<-apply(Twb.prof[ip.rows,], 2, mean)
fz.mean<-apply(Twb.prof[fz.rows,], 2, mean)
ip.var<-apply(Twb.prof[ip.rows,], 2, var)

plot(Twb.prof[1:100,], matrix(step.size,nrow=100,ncol=length(step.size),byrow=T),type='n', main='Mean IP vs FZRA Observation', xlab='Temperature', ylab='Meters AGL' )
lines(ip.mean, step.size, type='l', col='red')
lines(fz.mean, step.size, type='l', col='blue')
legend('topleft', c('FZRA', 'IP'), col=c('blue', 'red'), pch=19)
length(ip.rows)


#STATION LOCATIONS
par(mfrow=c(1,1))
library(maps)
library(mapdata)    #some additional hires data
library(maptools)   #useful tools such as reading shapefiles
library(mapproj)
length(stations) #551
lon2<-lon-360
par(mfrow=c(1,1))
plot(lon2, lat, main="Station Location", xlab="Longitude", ylab="Latitude", pch=18)
map("world",add=T)
map("state",add=T)

