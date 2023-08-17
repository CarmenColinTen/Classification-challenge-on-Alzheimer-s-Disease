library(ggplot2)
library(FNN)
library(class)
set.seed(1)
getwd()  #Current directory

setwd("C:/Users/carme/Desktop/UNICAS/STATISTIC/Challenge2")

#Read train and test data
train <- read.csv('train_ch.csv')
test <- read.csv('test_ch.csv')
test_complete <- read.csv('test_completo_ch.csv')

train <- train[2:11]

# 1st approach: LINEAR REGRESSION
## Scale predictors and responses

train_norm <- as.data.frame(scale(train[1:10]))
train_norm["Y"] <- list(train$Y)
  
##Fit linear model using all predictors and responses
linear_model <- lm( Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9, data=train_norm)

#Review the model and graphs
#Check P-value, for relation beetween the model or not in order to find if the predictors is contributing or not. 
summary(linear_model)
plot(linear_model)

#Check for correlation between predictors
tmp <- cor(train_norm)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0

#We can see there is high correlation in predictors V5, V1 and V7, we remove V1 and V7. V5 we keep it has a low P-Value
linear_model <- lm( Y ~ v2 + v3 + v4 + v5 + v6 + v8 + v9, data=train_norm)
summary(linear_model)

#We need to check for linearity and predictors with non-significant p-value
ggplot(train_norm, aes(v2, Y)) + 
  geom_point()

ggplot(train_norm, aes(v3, Y)) + 
  geom_point()
ggplot(train_norm, aes(v4, Y)) + 
  geom_point()
ggplot(train_norm, aes(v5, Y)) + 
  geom_point()
ggplot(train_norm, aes(v6, Y)) + 
  geom_point()
ggplot(train_norm, aes(v8, Y)) + 
  geom_point()
ggplot(train_norm, aes(v9, Y)) + 
  geom_point()

#Non-significant p-value and no linearity: V2, V4, V6, V8 and V9

linear_model <- lm( Y ~ v3 + v5, data=train_norm)
summary(linear_model)

par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

#Our graph represent a cuadratic function. We transform the predictors to see if that change the model
linear_model<- lm(Y ~ v5 + I(v5^2) + v3, data=train_norm)

par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))



#Final Model Linear regression
linear_model<- lm(Y ~ v3 + I(v3^2) + v5, data=train_norm)
par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

summary(linear_model)

#############################################################
#2nd Approach KNN

# Scaling Predictors. 
train_knn <- as.data.frame(scale(train[2:10]))

#We experiment with different numbers of K=0,1,2,3,4,5,6,7 etc 

train_knn <- data.frame(v3 = train$v3)

k_values <- c(1, 2, 3, 4, 5, 6, 7,8,9,10)  # Set the desired k values

mse_values <- numeric(length(k_values))  # Initialize vector to store MSE values

for (i in 1:length(k_values)) {
  k <- k_values[i]
  
  # Build the KNN model
  Knn_model <- knn.reg(train = train_knn, test = NULL, y = train$Y, k = k)
  
  # Calculate MSE
  mse_values[i] <- mean((Knn_model$residuals)^2)
}

mse_values


# We use V3 as in our linear regression Model. For k=5, not too big and good mse value
Knn_model <- knn.reg(train = train_knn$v3,  test = NULL, y=train$Y, k=5)
msqer <- mean((Knn_model$residuals)^2)
msqer

#Normalizing our test data according to train std and mean
mean_data_v3 <- mean(train$v3)
stdv_data_v3 <- sd(train$v3)
test_norm["v3"] <- (test$v3- mean_data_v3) / stdv_data_v3

mean_data_v5 <- mean(train$v5)
stdv_data_v5 <- sd(train$v5)
test_norm["v5"] <- (test$v5- mean_data_v5) / stdv_data_v5

#Fitting and predicting Ln and KNN
fit_lm <- lm(Y ~ v3 + I(v3^2) + v5, data=train_norm)
pred_lm <- predict(fit_lm, test_norm)

fit_knn <- knn.reg(train = as.data.frame(train_norm$v3), test=as.data.frame(test_norm$v3), y = train_norm$Y, k = 5)
pred_knn <- fit_knn$pred

# Combine the predicted values into a data frame
predictions <- data.frame(pred_lm, pred_knn)


#Calculate mse for each method
mylabel <- as.numeric(test_complete$Y)
mse_lm <- (pred_lm - (mylabel))^2
msqer_lm <- mean(mse_lm)
mse_knn <- (pred_knn - (mylabel))^2
msqer_knn <- mean(mse_knn)

print(msqer_lm)
print(msqer_knn)


# Save the data frame as a CSV file
write.csv(predictions, file = "final_predictions.csv", row.names = FALSE)

