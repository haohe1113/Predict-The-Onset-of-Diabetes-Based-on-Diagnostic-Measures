# Diabetes Prediction Based on Diagnostic Measures  
![intro](diabetes_intro.jpeg)  
## Read Data and Data Cleaning  

    ## Loading required package: Matrix

    ## Loaded glmnet 4.0-2

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'lattice'

    ## The following object is masked from 'package:boot':
    ## 
    ##     melanoma

    ## Loading required package: ggplot2

First we read in the data in which the zero values have all been
converted to NA values.Then we omit the observations which contain the
NA values.

``` r
rm(list=ls())
#setwd('/Users/z/Documents/MSBA/Summer 2020/Predictive Modeling/Project')
diabetes=read.csv("pima_diabetes_NA.csv")
diabetes$class <- as.factor(diabetes$class)
diabetes <- na.omit(diabetes)
attach(diabetes)
```

Next we are going to see if we have a class imbalance issue. The
percentage of true class which means the patient has diabetes is 33.61%
while the false class is 66.84%. There is imbalance existing in the
dataset which might impact our analysis.

``` r
D = sum(diabetes$class == 1)
TrueD =   D / nrow(diabetes)
TrueD
```

    ## [1] 0.3489583

``` r
1-TrueD
```

    ## [1] 0.6510417

We plotted the box plot to get a sense of what might be the key
variables to help us tell the difference between patients with diabetes
and patients without.In the plot we can see plas, mass, and age is doing
a better job in differentiating two classes.

``` r
par(mfrow=c(2,4)) 

plot(preg ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(plas ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(pres ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(skin ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(test ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(mass ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(pedi ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
plot(age ~ class, data=diabetes, col=c(grey(.2),2:6), cex.lab=1)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/fig1-1.png)

## Model Selection  

**KNN**  

We are going to perform a 5-fold CV KNN model here. We are testing the K
value for KNN ranging from 1 to 100. The optimal K is 17 which gives us
an accuracy of 77.44%.

``` r
par(mfrow=c(1,1))
plot(kk,mAccuracy,
     xlab="k",
     ylab="CV Accuracy",
     col=4, #Color of line
     lwd=2, #Line width
     type="l", #Type of graph = line
     cex.lab=1.2, #Size of labs
     main=paste("kfold(",kcv,")")) #Title of the graph

#Find the index of the minimum value of mMSE
best = which.max(mAccuracy)
text(kk[best],mAccuracy[best], #Coordinates
     paste("k=",kk[best]), #The actual text
     col=2, #Color of the text
     cex=0.8) #Size of the text
text(kk[best]+5,mAccuracy[best]-3,paste("Best Accuracy:",mAccuracy[best]), col=2, cex=0.8)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/fig2-1.png)

``` r
maxAccuracy = max(mAccuracy)
```

**Tree Model**  

We tried Simple Tree Model and Random Forest in this part.  

**Simple Tree**  

We grew the Simple Tree model based on one set up of train and test data
following the grow-and-prune process. Because of the randomness of train
and test data, **the accuracy rate of our Simple Tree model might floats**.

``` r
library(tree)
set.seed(1)
pima = read.csv("pima_diabetes.csv",header=T)

#Data Cleanning
pima$class = as.factor(pima$class)
pima[,2:8][pima[,2:8] == 0] = NA
pima = na.omit(pima)

#set up train and test data
n = dim(pima)[1]
train = sample(1:n, size = 130, replace = FALSE) #Indice of train data
pima_tr = pima[train, ] #the train data
pima_te = pima[-train, ] # the test data

#First, grow a big tree

big_tree = tree(class ~ ., data = pima_tr)
plot(big_tree)
text(big_tree)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
#make prediction on test data
pima_pred = predict(big_tree,
                    newdata = pima_te,
                    type="class") 

#compare predictions to actual diabetes status
conf_mtx = table(pima_pred,pima_te$class) #diagonal are correct predictions, off-diagonal are incorrect
conf_mtx
```

    ##          
    ## pima_pred   0   1
    ##         0 141  30
    ##         1  35  56

``` r
accur_big = sum(diag(conf_mtx)) / dim(pima_te)[1]
cat("we have a big tree with accuracy rate of", accur_big, '\n')
```

    ## we have a big tree with accuracy rate of 0.7519084

``` r
#Second, we prune back our big tree to reach the optimal size (cp)

#Do a 10-fold CV on our big tree
tree_cv = cv.tree(big_tree, 
                  FUN = prune.misclass,
                  K = 10)

#find the size# with min missclassification rate
plot(tree_cv, type = 'p')
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-5-2.png)

``` r
best_size = tree_cv$size[which.min(tree_cv$dev)]

#prune back the big tree to the optimum size
pruned_tree = prune.misclass(big_tree, best = best_size)
plot(pruned_tree)
text(pruned_tree, pretty = T)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-5-3.png)

``` r
#look at predictions
pruned_tree_pred = predict(pruned_tree, 
                            newdata = pima_te,
                            type="class")
conf_mtx = table(pruned_tree_pred, pima_te$class) 
conf_mtx #diagonal are correct predictions, off-diagonal are incorrect
```

    ##                 
    ## pruned_tree_pred   0   1
    ##                0 171  59
    ##                1   5  27

``` r
accur_pruned = sum(diag(conf_mtx)) / dim(pima_te)[1]
cat("we have a pruned tree with accuracy rate of", accur_pruned,
    "with size of", best_size, '\n')
```

    ## we have a pruned tree with accuracy rate of 0.7557252 with size of 3

**Random Forest**  

We then built the Random Forest model based on a 10-fold CV where we
tested the p value b/w 100 and 500 and the m value from 1 to 8 (total).
We ended up having an optimal RF with **m = 4** and **p = 100** which
yeilds an accuracy rate of **77.55%**.

``` r
set.seed(1)
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
pima = read.csv("pima_diabetes_NA.csv",header=T)

#Data Cleanning
pima$class = as.factor(pima$class)
pima[,2:8][pima[,2:8] == 0] = NA
pima = na.omit(pima)

#total num of obs in pima
n = dim(pima)[1]

#Define the number of folds, aka num of groups that the data is divided into
kcv = 10 

#Size of the fold (which is the number of elements in the test matrix)
n0 = round(n/kcv, 0) 

#The model setups matrix
p = ncol(pima) - 1 #Number of covariates (-1 because one column is the response)
mtryv = c(1:p) #Number of candidate variables for each split, the m val
ntreev = c(100,500) #Number of trees, the B val
parmrf = expand.grid(mtryv,ntreev) #Expading grids of different models
colnames(parmrf)=c('mtry','ntree')
nset = nrow(parmrf) #Number of models
#print(parmrf)

#error matrix
out_ERR = matrix(0, #matrix filled with zeroes
                 nrow = kcv, #number of rows
                 ncol = nset) #number of columns/models

#Vector of indices that have already been used inside the for
used = NULL

#The set of indices not used (will be updated removing the used)
set = 1:n

for(j in 1:kcv){
    
    if(n0 < length(set)) { #If the set of 'not used' is > than the size of the fold
        val = sample(set, size = n0) #then sample indices from the set
        #recall k-flod RADOMLY select obs from data to fill in each fold
    }
    if(n0 >= length(set)) { #If the set of 'not used' is <= than the size of the fold
        val = set #then use all of the remaining indices as the sample
    }
    
    #Create the train and test matrices
    train_i = pima[-val,] #Every observation except the ones whose indices were sampled
    test_i = pima[val,] #The observations whose indices sampled
    
    for(i in 1:nset){
        #The current model
        temprf = randomForest(class ~ .,
                              data = train_i,
                              mtry = parmrf[i,1],
                              ntree = parmrf[i,2],
                              maxnodes = 15)
        pred = predict(temprf,
                       newdata = test_i,
                       type = 'class')
        error_rate = sum(test_i$class != pred) / nrow(test_i)
        #Store the current ERR
        out_ERR[j,i] = error_rate
    }
    #The union of the indices used currently and previously, or, append val to used
    used = union(used,val)
    #The set of indices not used is updated
    set = (1:n)[-used]
    #Printing on the console the information that you want
    #Useful to keep track of the progress of your loop
    cat(j,"folds out of",kcv,'\n')
}
```

    ## 1 folds out of 10 
    ## 2 folds out of 10 
    ## 3 folds out of 10 
    ## 4 folds out of 10 
    ## 5 folds out of 10 
    ## 6 folds out of 10 
    ## 7 folds out of 10 
    ## 8 folds out of 10 
    ## 9 folds out of 10 
    ## 10 folds out of 10

``` r
#Calculate the mean of error rate for each model
mERR = apply(out_ERR, 2, mean) 

#plot m vs error rate graph for p = 100 and p = 500
plot(x = 1:8,
     y = sqrt(mERR[1:8]),
     xlab="m value",
     ylab="out-of-sample error rate",
     col=4, #Color of line
     lwd=2, #Line width
     type="l", #Type of graph = line
     cex.lab=1.2, #Size of labs
     main=paste("10-fold CV")) #Title of the graph
lines(x = 1:8,
     y = sqrt(mERR[9:16]),
     col = 5,
     lwd = 2)
legend("topright",
       legend=c("p = 100", "p = 500"),
       col=c(4, 5),
       lty=c(1, 1),
       cex=0.8)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
#pick up the best model overall 
best = which.min(mERR)
bestrf = randomForest(class ~ .,
                      data = pima,
                      mtry = parmrf[best,1],
                      ntree = parmrf[best,2],
                      maxnodes = 15)

#calculate accuracy rate when predicting the whole population for pima
bestrfpred = predict(bestrf, type="class")
conf_mtx = table(bestrfpred, pima$class) 
#conf_mtx #diagonal are correct predictions, off-diagonal are incorrect
accur_rf = sum(diag(conf_mtx)) / dim(pima)[1]
cat("we have a random forest model with accuracy rate of", accur_rf,
    "with m =", parmrf[best, 1],
    "p = ", parmrf[best, 2], '\n')
```

    ## we have a random forest model with accuracy rate of 0.7755102 with m = 4 p =  100

**Logistic Regression**  

In the end we performed a Logistic Regression with cross validation. The
raw CV estimate of accuracy returned is **78.06%** and the **adjusted**
estimate is **77.60%**.

``` r
# first we scale the data that will be used later for variable selection 
XXdia <- scale(diabetes[,-9])
DIAdata <- data.frame(class,XXdia)
#define our own cost function here
mycost <- function(r, pi){  #r = observed responses, pi = fitted responses
  c1 = (r==1)&(pi<0.5) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi>=0.5) #logical vector - true if actual 0 but predict 1
  return(mean(c1+c0))
}
diabetes.glm <- glm(class ~ ., family = 'binomial', data = DIAdata)
set.seed(1)
cv.error = cv.glm(DIAdata,diabetes.glm, cost = mycost, K=10)$delta
cv.error  
```

    ## [1] 0.2278646 0.2255198

## Variable Selection  

As previously mentioned with the box plot, we notice some variables
might have stronger predictive ability than others. Here,we will perform
out-of-sample variable selection methods with Logistic Regression
algorithm to find out the more important predictors.

``` r
#split the dataset into training and test data for all the variable selection methods for better comparison
n = dim(diabetes)[1]
set.seed(1)
tr = sample(1:n, 
             size = 274, # we take 70% of our dataset as training data
             replace = FALSE) 
```

### Out-of-Sample Stepwise Methods  

We tried all three kinds of stepwise method: forward, backward and both.
The results returned are the same with the model **class \~ plas + mass
+ age** and an accuracy of **76.27%**.

**Forward Selection**  

``` r
test_x = DIAdata[-tr,][,-1]
test_y = DIAdata[-tr,][,1]

#Two models initially
null = glm(class~1, data=DIAdata[tr,], family = 'binomial' ) 
full = glm(class~., data=DIAdata[tr,], family = 'binomial') 

regForward = step(null, 
                  scope=formula(full),
                  direction="forward", 
                  k=log(length(tr))) 
```

    ## Start:  AIC=351.1
    ## class ~ 1
    ## 
    ##        Df Deviance    AIC
    ## + plas  1   302.24 313.46
    ## + mass  1   321.63 332.86
    ## + preg  1   332.51 343.74
    ## + pedi  1   332.87 344.09
    ## + age   1   333.55 344.78
    ## <none>      345.49 351.10
    ## + test  1   344.00 355.23
    ## + skin  1   344.18 355.41
    ## + pres  1   345.45 356.67
    ## 
    ## Step:  AIC=313.46
    ## class ~ plas
    ## 
    ##        Df Deviance    AIC
    ## + mass  1   284.90 301.74
    ## + preg  1   290.95 307.79
    ## + pedi  1   293.52 310.36
    ## <none>      302.24 313.46
    ## + age   1   296.93 313.77
    ## + test  1   300.53 317.37
    ## + pres  1   300.93 317.77
    ## + skin  1   302.09 318.93
    ## 
    ## Step:  AIC=301.74
    ## class ~ plas + mass
    ## 
    ##        Df Deviance    AIC
    ## + preg  1   271.94 294.39
    ## + age   1   273.28 295.73
    ## + test  1   279.22 301.67
    ## <none>      284.90 301.74
    ## + pedi  1   279.67 302.12
    ## + skin  1   282.55 305.01
    ## + pres  1   282.82 305.27
    ## 
    ## Step:  AIC=294.39
    ## class ~ plas + mass + preg
    ## 
    ##        Df Deviance    AIC
    ## + pedi  1   265.90 293.97
    ## <none>      271.94 294.39
    ## + pres  1   266.93 294.99
    ## + test  1   268.35 296.41
    ## + age   1   268.70 296.77
    ## + skin  1   270.20 298.27
    ## 
    ## Step:  AIC=293.97
    ## class ~ plas + mass + preg + pedi
    ## 
    ##        Df Deviance    AIC
    ## + test  1   259.04 292.72
    ## + pres  1   259.96 293.64
    ## <none>      265.90 293.97
    ## + skin  1   262.48 296.16
    ## + age   1   262.68 296.36
    ## 
    ## Step:  AIC=292.72
    ## class ~ plas + mass + preg + pedi + test
    ## 
    ##        Df Deviance    AIC
    ## + pres  1   252.88 292.18
    ## <none>      259.04 292.72
    ## + age   1   256.89 296.19
    ## + skin  1   258.40 297.69
    ## 
    ## Step:  AIC=292.18
    ## class ~ plas + mass + preg + pedi + test + pres
    ## 
    ##        Df Deviance    AIC
    ## <none>      252.88 292.18
    ## + age   1   249.81 294.71
    ## + skin  1   252.46 297.37

``` r
forward_prob <- predict(regForward, test_x, type = "response")
forward_pred <- ifelse(forward_prob > 0.5, 1, 0)
forward_acc = mean(forward_pred==test_y)

forward_acc
```

    ## [1] 0.7651822

**Backward Selection**  

``` r
regBack = step(full, 
               direction="backward", 
               k=log(length(tr))) 
```

    ## Start:  AIC=299.99
    ## class ~ preg + plas + pres + skin + test + mass + pedi + age
    ## 
    ##        Df Deviance    AIC
    ## - skin  1   249.81 294.71
    ## - age   1   252.46 297.36
    ## - test  1   253.25 298.16
    ## <none>      249.48 299.99
    ## - preg  1   256.13 301.03
    ## - pres  1   256.36 301.27
    ## - pedi  1   259.85 304.75
    ## - mass  1   273.10 318.01
    ## - plas  1   284.43 329.34
    ## 
    ## Step:  AIC=294.71
    ## class ~ preg + plas + pres + test + mass + pedi + age
    ## 
    ##        Df Deviance    AIC
    ## - age   1   252.88 292.18
    ## <none>      249.81 294.71
    ## - test  1   255.53 294.82
    ## - preg  1   256.34 295.63
    ## - pres  1   256.89 296.19
    ## - pedi  1   259.91 299.20
    ## - mass  1   274.32 313.61
    ## - plas  1   285.21 324.50
    ## 
    ## Step:  AIC=292.18
    ## class ~ preg + plas + pres + test + mass + pedi
    ## 
    ##        Df Deviance    AIC
    ## <none>      252.88 292.18
    ## - pres  1   259.04 292.72
    ## - test  1   259.96 293.64
    ## - pedi  1   263.39 297.07
    ## - preg  1   267.05 300.73
    ## - mass  1   274.57 308.25
    ## - plas  1   294.88 328.56

``` r
back_prob <- predict(regBack, test_x, type = "response")
back_pred <- ifelse(back_prob > 0.5, 1, 0)
back_acc = mean(back_pred==test_y)

back_acc
```

    ## [1] 0.7651822

**Both Selection**  

``` r
regHybrid = step(null, 
                 scope=formula(full), 
                 direction="both", 
                 k=log(length(tr))) 
```

    ## Start:  AIC=351.1
    ## class ~ 1
    ## 
    ##        Df Deviance    AIC
    ## + plas  1   302.24 313.46
    ## + mass  1   321.63 332.86
    ## + preg  1   332.51 343.74
    ## + pedi  1   332.87 344.09
    ## + age   1   333.55 344.78
    ## <none>      345.49 351.10
    ## + test  1   344.00 355.23
    ## + skin  1   344.18 355.41
    ## + pres  1   345.45 356.67
    ## 
    ## Step:  AIC=313.46
    ## class ~ plas
    ## 
    ##        Df Deviance    AIC
    ## + mass  1   284.90 301.74
    ## + preg  1   290.95 307.79
    ## + pedi  1   293.52 310.36
    ## <none>      302.24 313.46
    ## + age   1   296.93 313.77
    ## + test  1   300.53 317.37
    ## + pres  1   300.93 317.77
    ## + skin  1   302.09 318.93
    ## - plas  1   345.49 351.10
    ## 
    ## Step:  AIC=301.74
    ## class ~ plas + mass
    ## 
    ##        Df Deviance    AIC
    ## + preg  1   271.94 294.39
    ## + age   1   273.28 295.73
    ## + test  1   279.22 301.67
    ## <none>      284.90 301.74
    ## + pedi  1   279.67 302.12
    ## + skin  1   282.55 305.01
    ## + pres  1   282.82 305.27
    ## - mass  1   302.24 313.46
    ## - plas  1   321.63 332.86
    ## 
    ## Step:  AIC=294.39
    ## class ~ plas + mass + preg
    ## 
    ##        Df Deviance    AIC
    ## + pedi  1   265.90 293.97
    ## <none>      271.94 294.39
    ## + pres  1   266.93 294.99
    ## + test  1   268.35 296.41
    ## + age   1   268.70 296.77
    ## + skin  1   270.20 298.27
    ## - preg  1   284.90 301.74
    ## - mass  1   290.95 307.79
    ## - plas  1   307.07 323.91
    ## 
    ## Step:  AIC=293.97
    ## class ~ plas + mass + preg + pedi
    ## 
    ##        Df Deviance    AIC
    ## + test  1   259.04 292.72
    ## + pres  1   259.96 293.64
    ## <none>      265.90 293.97
    ## - pedi  1   271.94 294.39
    ## + skin  1   262.48 296.16
    ## + age   1   262.68 296.36
    ## - preg  1   279.67 302.12
    ## - mass  1   281.06 303.52
    ## - plas  1   297.68 320.14
    ## 
    ## Step:  AIC=292.72
    ## class ~ plas + mass + preg + pedi + test
    ## 
    ##        Df Deviance    AIC
    ## + pres  1   252.88 292.18
    ## <none>      259.04 292.72
    ## - test  1   265.90 293.97
    ## + age   1   256.89 296.19
    ## - pedi  1   268.35 296.41
    ## + skin  1   258.40 297.69
    ## - preg  1   269.83 297.90
    ## - mass  1   278.11 306.17
    ## - plas  1   297.47 325.54
    ## 
    ## Step:  AIC=292.18
    ## class ~ plas + mass + preg + pedi + test + pres
    ## 
    ##        Df Deviance    AIC
    ## <none>      252.88 292.18
    ## - pres  1   259.04 292.72
    ## - test  1   259.96 293.64
    ## + age   1   249.81 294.71
    ## - pedi  1   263.39 297.07
    ## + skin  1   252.46 297.36
    ## - preg  1   267.05 300.73
    ## - mass  1   274.57 308.25
    ## - plas  1   294.88 328.56

``` r
hybrid_prob <- predict(regHybrid, test_x, type = "response")
hybrid_pred <- ifelse(hybrid_prob > 0.5, 1, 0)
hybrid_acc = mean(hybrid_pred==test_y)

hybrid_acc
```

    ## [1] 0.7651822

### Out-of-Sample Shrinkage Method  

Lasso returns with an accuracy of **77.11%** with the model
**class\~plas+mass+age**.By looking at the coefficients, plas has a more
significant predicting power, mass comes the second, and age comes the
third. This verifies our guess when first looking at the box plot. And
the predictors chosen by Lasso is also consistent with what we get from
the stepwise method.

Ridge, on the other hand, has a compromised performance with an accuracy
of **72.88%** only.

**Lasso**  

``` r
train_x = as.matrix(DIAdata[tr,-1])
train_y = DIAdata[tr,1] 
test_x = as.matrix(DIAdata[-tr,-1])
test_y = DIAdata[-tr,1] 

Lasso.Fit = glmnet(train_x,train_y,family = 'binomial',alpha=1)
plot(Lasso.Fit)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-12-1.png)

``` r
#Cross Validation
set.seed(1)
CV.L = cv.glmnet(train_x, train_y, family = 'binomial',type.measure="class", alpha=1) 
coef(CV.L,s="lambda.1se")
```

    ## 9 x 1 sparse Matrix of class "dgCMatrix"
    ##                      1
    ## (Intercept) -0.9314809
    ## preg         0.3155049
    ## plas         0.8844288
    ## pres        -0.2435930
    ## skin         .        
    ## test        -0.1872814
    ## mass         0.6453351
    ## pedi         0.3326329
    ## age          0.2289776

``` r
####Plots#####
LamL = CV.L$lambda.1se
plot(log(CV.L$lambda),sqrt(CV.L$cvm),
     main="LASSO CV (k=10)",xlab="log(lambda)",
     ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(LamL),lty=2,col=2,lwd=2)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-12-2.png)

``` r
####Predict on test set####
lasso_predict = predict(CV.L,newx = test_x,s="lambda.1se",type='class')
lasso_conf_matrix <- table(lasso_predict,test_y)
lasso_acc = mean(lasso_predict==test_y)
lasso_acc
```

    ## [1] 0.7753036

**Ridge**    

``` r
Ridge.Fit = glmnet(train_x,train_y,family = 'binomial',alpha=0)
par(mfrow=c(1,1)) 
plot(Ridge.Fit)
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-13-1.png)

``` r
#Cross Validation
set.seed(1)
CV.R = cv.glmnet(train_x, train_y,family = 'binomial',type.measure="class", alpha=0)
coef(CV.R,s="lambda.1se") # check coefficients of the model with optimal lambda
```

    ## 9 x 1 sparse Matrix of class "dgCMatrix"
    ##                       1
    ## (Intercept) -0.85131850
    ## preg         0.23895294
    ## plas         0.57684643
    ## pres        -0.16125025
    ## skin        -0.02002929
    ## test        -0.08353884
    ## mass         0.44403301
    ## pedi         0.26467225
    ## age          0.21115009

``` r
#this lambda is not correponding to the lowest RMSE, instead its error is within one std of the min RMSE

#Make the plot of how cv chooses the optimal lambda
LamR = CV.R$lambda.1se
plot(log(CV.R$lambda),sqrt(CV.R$cvm),
     main="Ridge CV (k=10)",
     xlab="log(lambda)",
     ylab = "RMSE",
     col=4,#Color of points
     cex.lab=1.2) #Size o lab
abline(v=log(LamR),lty=2,col=2,lwd=2) #selected lambda vs RMSE
```

![](FInal_Project_ver1.3_files/figure-markdown_github/unnamed-chunk-13-2.png)

``` r
#predict on the test data
ridg_predict = predict(CV.R, newx = test_x,s="lambda.1se",type='class')
ridg_acc = mean(ridg_predict==test_y)
ridg_conf_matrix <- table(ridg_predict,test_y)

ridg_acc
```

    ## [1] 0.7651822

## Model Comparison  

Most of our models yield satisfactory accuracy of above 75%. We want to
specifically compare the Logistic Regression Model with all variables
and Lasso with three variables picked to evalute whether variable
selection process is effective. Both models have similar accuracy, while
our team decides to go with the Logistic Regression with all variables.

The main reason is that our model is trying to solve the problem of
predicting diabetes. It is crucial to correctly identify the patients
with diabetes for timely treatment. Therefore, sensitivity score of the
model is an important criteria to look at besides the accuracy score.

Below we are doing the out-of-sample Logistic Regression in order to
better compare with the Lasso model. We will be also generating the
confusion matrix for both models.

``` r
#
train = DIAdata[tr,] 
test_x = DIAdata[-tr,][,-1]
test_y = DIAdata[-tr,][,1]

diabetes.glm <- glm(class ~ ., family = 'binomial', data = train)
glm_predict_prob <-predict(diabetes.glm, newdata = test_x, family = 'binomial',type = 'response')
glm_predict <- ifelse(glm_predict_prob > 0.5, 1, 0)
glm_predict_accuracy <- mean(glm_predict == test_y)


glm_conf_matrix <- table(glm_predict,test_y)
glm_conf_matrix
```

    ##            test_y
    ## glm_predict   0   1
    ##           0 282  81
    ##           1  33  98

``` r
#confusion matrix for lasso
lasso_conf_matrix
```

    ##              test_y
    ## lasso_predict   0   1
    ##             0 287  83
    ##             1  28  96

*Logistic Regression with all Variables* Sensitivity Score: **0.6154**
Specificity Score: **0.8734**

*Lasso with class\~plas+mass+age* Sensitivity Score: **0.4103**
Specificity Score: **0.9494**

As we can see, Logistic Regression with all variables achieve a leap in
the sensitivity score with a small compromise of specificity score. This
might be due to the information contained in the variables that aren’t
selected by Lasso.

What’s more, since our dataset only has 8 predictors, we are not
analyzing our data in a high-dimensional space which requires varaibles
selection methods to reduce the dimensionality.
