library(randomForest)
library(ggplot2)
library(gridExtra)
library(glmnet)

######################
##  Data prepration ##
######################

data= read.csv('C:/Users/xy249/Desktop/stockdata2018.csv')
head(data)
standard =function(x){
  x/sqrt(mean((x-mean(x))^2))
}
data1=apply(data, 2, standard)
head(data1)
summary(data1)
boxplot(X)
hist(X)
n        =    dim(data1)[1]
p        =    dim(data1)[2]-1
y        =    data1[,1]
X        =    data.matrix(data1[,-1])

##
###split datasets into testing set and training set.
##

n.train          =     floor(0.8*n)
n.test           =     n-n.train
M                =     100
Rsq.test.rf      =     rep(0,M)  # rf= randomForest
Rsq.train.rf     =     rep(0,M)
Rsq.test.el      =     rep(0,M)  #el = elastic net
Rsq.train.el     =     rep(0,M)
Rsq.test.rid     =     rep(0,M)
Rsq.train.rid    =     rep(0,M)
Rsq.test.lasso   =     rep(0,M) 
Rsq.train.lasso  =     rep(0,M)
time.rid         =     rep(0,M)
time.lasso       =     rep(0,M)
time.rf          =     rep(0,M)
time.el          =     rep(0,M)
############################################################
## Repeat each model 100 times and calculate running time ##
############################################################

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  start.time=Sys.time()
  #fit ridge and calculate and record the train and test R^2
  cv.rid            =     cv.glmnet(X.train, y.train, alpha = 0, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  rid.fit           =     glmnet(X.train, y.train, alpha = 0, family = "gaussian", intercept = T, lambda = cv.rid$lambda.min)
  y.train.hat.rid   =     X.train %*% rid.fit$beta + rid.fit$a0  
  y.test.hat.rid    =     X.test %*% rid.fit$beta  + rid.fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat.rid)^2)/mean((y - mean(y))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat.rid)^2)/mean((y - mean(y))^2)
  end.time=Sys.time()
  time.rid[m]=end.time-start.time
  
  
  start.time=Sys.time()
  #fit lasso and calculate and record the train and test R^2
  cv.lasso            =     cv.glmnet(X.train, y.train, alpha = 1, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  lasso.fit           =     glmnet(X.train, y.train, alpha = 1, family = "gaussian", intercept = T, lambda = cv.lasso$lambda.min)
  y.train.hat.lasso   =     X.train %*% lasso.fit$beta + lasso.fit$a0  
  y.test.hat.lasso    =     X.test %*% lasso.fit$beta  + lasso.fit$a0
  Rsq.test.lasso[m]   =     1-mean((y.test - y.test.hat.lasso)^2)/mean((y - mean(y))^2)
  Rsq.train.lasso[m]  =     1-mean((y.train - y.train.hat.lasso)^2)/mean((y - mean(y))^2)
  end.time=Sys.time()
  time.lasso[m]=end.time-start.time
  
  start.time=Sys.time()
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  cv.el            =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  el.fit           =     glmnet(X.train, y.train, alpha = a, lambda = cv.el$lambda.min)
  y.train.hat.el   =     predict(el.fit, newx = X.train, type = "response")
  y.test.hat.el    =     predict(el.fit, newx = X.test, type = "response")
  Rsq.test.el[m]   =     1-mean((y.test - y.test.hat.el)^2)/mean((y - mean(y))^2)
  Rsq.train.el[m]  =     1-mean((y.train - y.train.hat.el)^2)/mean((y - mean(y))^2)
  end.time=Sys.time()
  time.el[m]=end.time-start.time
  
  
  start.time=Sys.time()
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat.rf    =     predict(rf, X.test)
  y.train.hat.rf   =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat.rf)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat.rf)^2)/mean((y - mean(y))^2)
  end.time=Sys.time()
  time.rf[m]=end.time-start.time
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.el=%.2f| Rsq.train.rid=%.2f,  Rsq.train.lasso=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.el[m],  Rsq.train.rid[m], Rsq.train.lasso[m]))
  
}
cat(sprintf('Ridge regression runing time:%4f \nLasso regression running time:%4f \nEl regression running time:%4f \nRandom forest running time:%4f \n', sum(time.rid),sum(time.lasso),sum(time.el),sum(time.rf)))




###########################################################
## Boxplots of testingset and training set of each model ##
###########################################################

#boxplot rf
par(mfrow=c(1,2))
boxplot(Rsq.train.rf)
boxplot(Rsq.test.rf)

#boxplot el
boxplot(Rsq.train.el)
boxplot(Rsq.test.el)

#boxplot rid
boxplot(Rsq.train.rid)
boxplot(Rsq.test.rid)

#boxplot lasso
boxplot(Rsq.train.lasso)
boxplot(Rsq.test.lasso)



######################
## 10-fold CV curve ##
######################


for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(X.train, y.train, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

par(mfrow=c(3,1))

#may show margins too large error, expand plot area to fix it.
plot(fit10, main="LASSO")
plot(fit0, main="Ridge")
plot(fit5, main="Elastic Net")





##############################################################
## Residual boxplots of testing and training of each model. ##
##############################################################
#rid
re.test.rid=y.test - (X.test %*% rid.fit$beta  + rid.fit$a0)
re.train.rid=y.train - (X.train %*% rid.fit$beta +rid.fit$a0)
re.train.rid=re.train.rid[,1]
re.test.rid=re.test.rid[,1]
par(mfrow=c(1,2))
boxplot(re.test.rid)
boxplot(re.train.rid)

#lasso
re.test.lasso=y.test - (X.test %*% lasso.fit$beta  + lasso.fit$a0)
re.train.lasso=y.train - (X.train %*% lasso.fit$beta +lasso.fit$a0)
re.train.lasso=re.train.lasso[,1]
re.test.lasso=re.test.lasso[,1]
par(mfrow=c(1,2))
boxplot(re.test.lasso)
boxplot(re.train.lasso)

#el
re.test.el=y.test - (X.test %*% el.fit$beta  + el.fit$a0)
re.train.el=y.train - (X.train %*% el.fit$beta +el.fit$a0)
re.train.el=re.train.el[,1]
re.test.el=re.test.el[,1]
par(mfrow=c(1,2))
boxplot(re.test.el)
boxplot(re.train.el)

#rf
re.test.rf=y.test - y.test.hat.rf
re.train.rf=y.train - y.train.hat.rf

par(mfrow=c(1,2))
boxplot(re.test.rf)
boxplot(re.train.rf)



############################
## Barplot with bootstrap ##
############################

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.el.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.lasso.bs    =     matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs el
  a                =     0.5 # elastic-net
  cv.el            =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  el.fit           =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.el$lambda.min)  
  beta.el.bs[,m]   =     as.vector(el.fit$beta)
  # fit bs rid
  cv.rid           =     cv.glmnet(X.train, y.train, alpha = 0, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  rid.fit          =     glmnet(X.train, y.train, alpha = 0, family = "gaussian", intercept = T, lambda = cv.rid$lambda.min)
  beta.rid.bs[,m]  =     as.vector(rid.fit$beta)
  # fit bs lasso
  cv.lasso         =     cv.glmnet(X.train, y.train, alpha = 1, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  lasso.fit        =     glmnet(X.train, y.train, alpha = 1, family = "gaussian", intercept = T, lambda = cv.lasso$lambda.min)
  beta.lasso.bs[,m]=     as.vector(lasso.fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
el.bs.sd    = apply(beta.el.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")
lasso.bs.sd = apply(beta.lasso.bs, 1, "sd")


# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit el to the whole data
a=0.5 # elastic-net
cv.el            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
el.fit           =     glmnet(X, y, alpha = a, lambda = cv.el$lambda.min)

# fit rid to the whole data
cv.rid           =     cv.glmnet(X, y, alpha = 0, nfolds = 10)
rid.fit          =     glmnet(X, y, alpha = 0, lambda = cv.rid$lambda.min)
# fit lasso to the whole data
cv.lasso         =     cv.glmnet(X, y, alpha = 1, nfolds = 10)
lasso.fit        =     glmnet(X, y, alpha = 1, lambda = cv.lasso$lambda.min)



betaS.rf               =     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.el               =     data.frame(names(X[1,]), as.vector(el.fit$beta), 2*el.bs.sd)
colnames(betaS.el)     =     c( "feature", "value", "err")

betaS.rid              =     data.frame(names(X[1,]), as.vector(rid.fit$beta), 2*rid.bs.sd)
colnames(betaS.rid)    =     c( "feature", "value", "err")

betaS.lasso            =     data.frame(names(X[1,]), as.vector(lasso.fit$beta), 2*lasso.bs.sd)
colnames(betaS.lasso)  =     c( "feature", "value", "err")


# need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.el$feature     =  factor(betaS.el$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rid$feature[order(betaS.rid$value, decreasing = TRUE)])
betaS.lasso$feature  =  factor(betaS.lasso$feature, levels = betaS.lasso$feature[order(betaS.lasso$value, decreasing = TRUE)])


rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Rf importance of variables')

elPlot =  ggplot(betaS.el, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('El importance of variables')

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Rid importance of variables')

lassoPlot =  ggplot(betaS.lasso, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Lasso importance of variables')


grid.arrange(rfPlot, elPlot, ridPlot,lassoPlot,nrow = 4)



