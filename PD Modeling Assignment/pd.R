### Initial Steps -----------------------------------------------------------------------
# load relevant libraries
library(data.table)
library(caTools)
library(corrplot)
library(ROCR)

# set path
path <- "F:/CRISIL/PD Modeling"


# load the data
dat <- fread(paste(path,'/Loan_Data.csv', sep = ''))

# summarize the data
summary(dat) # also shows which variable has NAs


### Variable Sanitization ---------------------------------------------------------------
# print out the variable classes and see which ones need changing
# ...notice how int_rate and revol_util are read as 'character'
# ...so are grade and emp_length (which need to be converted to factor)
data.frame(variable = names(dat),
           variable_type = as.character(sapply(dat, class)))
# make certain variables 'factor' (categorical)
dat$Default <- factor(dat$Default)
dat$grade <- factor(dat$grade)
dat$emp_length <- factor(dat$emp_length)
dat$delinq_2yrs <- factor(dat$delinq_2yrs)
dat$inq_last_6mths <- factor(dat$inq_last_6mths)

# convert the '%' character variables to numeric
dat$int_rate <- as.numeric(sub('%','',dat$int_rate))/100
dat$revol_util <- as.numeric(sub('%','',dat$revol_util))/100

# check the variable classes now...
data.frame(variable = names(dat),
           variable_type = as.character(sapply(dat, class)))

### Missing Value Treatment -------------------------------------------------------------

# Check fraction of each variable having missing values
data.frame(variable = names(dat),
           pctg_missing = 100*as.numeric(sapply(dat, 
                                                function(x) sum(is.na(x))/
                                                  length(x))))

# ...indicating we can remove mths_since_last_delinq which has > 60% NAs
dat$mths_since_last_delinq <- NULL

# distribution of revol_util variable before mean imputation:
plot(density(dat$revol_util, na.rm = T))
# check whether mean imputation causes spikes in distribution of revol_util
x <- dat$revol_util; x[is.na(x)] <- mean(x, na.rm = T); plot(density(x))
# no appreciablechange in density; no spikes; proceed with mean imputation
dat$revol_util[is.na(dat$revol_util)] <- mean(dat$revol_util, na.rm = T)

# dealing with n/a in emp_length
summary(dat$emp_length); levels(dat$emp_length)
dat$emp_length[dat$emp_length == 'n/a'] <- names(which.max(summary(dat$emp_length)))

# summarize and have a look at levels in emp_length once more:
summary(dat$emp_length); levels(dat$emp_length)
# remove the n/a level from the variable altogether:
dat$emp_length <- factor(dat$emp_length)
summary(dat$emp_length); levels(dat$emp_length)

### Variable Transformations ------------------------------------------------------------
# loan_amnt:
plot(density(dat$loan_amnt))
plot(density(sqrt(dat$loan_amnt))) # makes it 'more normal' let's verify:
qqnorm(dat$loan_amnt, pch = 20); qqline(dat$loan_amnt, col = 'red', lwd = 2)
qqnorm(dat$loan_amnt^0.5, pch = 20); qqline(dat$loan_amnt^0.5, col = 'red', lwd = 2)
# ...therefore:
dat$loan_amnt <- sqrt(dat$loan_amnt)

# open_acc:
plot(density(dat$open_acc))
plot(density(log(dat$open_acc))) # again, this makes it 'more normal'
qqnorm(dat$open_acc, pch = 20); qqline(dat$open_acc, col = 'red', lwd = 2)
qqnorm(log(dat$open_acc), pch = 20); qqline(log(dat$open_acc), col = 'red', lwd = 2)
# ...therefore:
dat$open_acc <- log(dat$open_acc)

# revol_bal
plot(density(dat$revol_bal))
plot(density(dat$revol_bal^0.3)) # this transformation makes revol_bal 'more normal'
qqnorm(dat$revol_bal, pch = 20); qqline(dat$revol_bal, col = 'red', lwd = 2)
qqnorm(dat$revol_bal^0.3, pch = 20); qqline(dat$revol_bal^0.3, col = 'red', lwd = 2)
# ...clearly, it's better to transform this variable as follows:
dat$revol_bal <- dat$revol_bal^0.3

# total_pymnt
plot(density(dat$total_pymnt))
plot(density(sqrt(dat$total_pymnt))) # which looks 'more normal'
qqnorm(dat$total_pymnt, pch = 20); qqline(dat$total_pymnt, col = 'red', lwd = 2)
qqnorm(dat$total_pymnt^0.5, pch = 20); qqline(dat$total_pymnt^0.5, col = 'red', lwd = 2)
# ...clearly, it's better to transform this variable with square root
dat$total_pymnt <- sqrt(dat$total_pymnt)

### Splitting the Data ------------------------------------------------------------------
set.seed(1)
spl <- sample.split(dat$Default, SplitRatio = 0.7)
train <- subset(dat, spl == T); test <- subset(dat, spl == F)

### Models ------------------------------------------------------------------------------
# Model 1 (All Variables) --------------------------------------------------------------
logitModel_01 <- glm(Default ~ ., data = train, family = 'binomial')
logit_01 <- summary(logitModel_01)

## Training Set Performance ----------------------------------------
predictions_train_01 <- predict(logitModel_01, newdata = train, type = 'response')
ROC_train_01 <- performance(prediction(predictions_train_01, 
                                       train$Default),
                            'tpr','fpr')
# Plot the ROC curve for Model 1 on train data
plot(ROC_train_01, colorize = T)
# Model 1 AUC on train data
auc_train_01 <- performance(prediction(predictions_train_01,
                                       train$Default),
                            'auc')@y.values[[1]]
# Print Model 1 AUC on train data
auc_train_01 # 0.9489708

# Overall Accuracy, Sensitivity, Specificity
cutoff <- 0.2 # choose cutoff that maximizes TPR and minimizes FPR
accuracy_train_01 <- (table(train$Default, predictions_train_01 > cutoff)[1,1] + 
                        table(train$Default, predictions_train_01 > cutoff)[2,2]) / 
  sum(sum(table(train$Default, predictions_train_01 > cutoff)))
sensitivity_train_01 <- table(train$Default, predictions_train_01 > cutoff)[2,2] / 
  sum(table(train$Default, predictions_train_01)[2,])
specificity_train_01 <- table(train$Default, predictions_train_01 > cutoff)[1,1] / 
  sum(table(train$Default, predictions_train_01)[1,])
# print out overall accuracy, sensitivity and specificity
paste('training set accuracy: ', round(accuracy_train_01, 2), sep = ''); paste('training set sensitivity: ', round(sensitivity_train_01, 2), sep = ''); paste('training set specificity: ', round(specificity_train_01, 2), sep = '')

## Test Set Performance --------------------------------------------
predictions_test_01 <- predict(logitModel_01, newdata = test, type = 'response')
ROC_test_01 <- performance(prediction(predictions_test_01,
                                      test$Default),
                           'tpr','fpr')
# Plot the ROC curve for Model 1 on test data
plot(ROC_test_01, colorize = T)
# Model 1 AUC on test data
auc_test_01 <- performance(prediction(predictions_test_01,
                                      test$Default),
                           'auc')@y.values[[1]]
# Print Model 1 AUC on test data
auc_test_01 # 0.9459849

# Overall Accuracy, Sensitivity, Specificity
cutoff <- 0.2 # choose cutoff that maximizes TPR and minimizes FPR
accuracy_test_01 <- (table(test$Default, predictions_test_01 > cutoff)[1,1] + 
                        table(test$Default, predictions_test_01 > cutoff)[2,2]) / 
  sum(sum(table(test$Default, predictions_test_01 > cutoff)))
sensitivity_test_01 <- table(test$Default, predictions_test_01 > cutoff)[2,2] / 
  sum(table(test$Default, predictions_test_01)[2,])
specificity_test_01 <- table(test$Default, predictions_test_01 > cutoff)[1,1] / 
  sum(table(test$Default, predictions_test_01)[1,])
# print out overall accuracy, sensitivity and specificity
paste('testing set accuracy: ', round(accuracy_test_01, 2), sep = ''); paste('testing set sensitivity: ', round(sensitivity_test_01, 2), sep = ''); paste('testing set specificity: ', round(specificity_test_01, 2), sep = '')

