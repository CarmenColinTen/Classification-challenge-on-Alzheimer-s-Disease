# Load necessary libraries
library(ggplot2)
library(leaps)
library(glmnet)
library(pROC)
library(caret)

library(MASS)
# Set seed for reproducibility
set.seed(2)

# Set working directory
setwd("C:/Users/carme/Desktop/UNICAS/STATISTIC/Challenge1")

# Read train and test data
train_data <- read.csv('ADCTLtrain.csv')
test_data <- read.csv('ADCTLtest.csv')
ids <- train_data[1]

# Convert CTL, AD to 0, 1
train_data$Label <- ifelse(train_data$Label == "CTL", 0, 1)
labels <- train_data$Label

# Extract features
features <- train_data[2:430]

# Preprocessing: Scaling features
scaled_features <- as.data.frame(scale(features))

# Check correlation between features
cor_matrix <- cor(scaled_features)
cor_matrix[upper.tri(cor_matrix)] <- 0
diag(cor_matrix) <- 0

# Remove highly correlated features
uncorrelated_features <- scaled_features[, !apply(cor_matrix, 2, function(x) any(x >= 0.7 & x <= 0.99, na.rm = FALSE))]
uncorrelated_features$label <- train_data$Label

# Feature Selection: Hybrid Forward and Backward
regfit.fwd <- regsubsets(label ~ ., data = uncorrelated_features, nvmax = 200, method = "forward")
regfit.bwd <- regsubsets(label ~ ., data = uncorrelated_features, nvmax = 200, method = "backward")

# Select best predictors from forward selection
fwd.max_adjr2 <- which.max(summary(regfit.fwd)$adjr2)
model.fwd_adjr2 <- summary(regfit.fwd)$which[fwd.max_adjr2, -1]

fwd.min_cp <- which.min(summary(regfit.fwd)$cp)
model.fwd_cp <- summary(regfit.fwd)$which[fwd.min_cp, -1]

fwd.min_bic <- which.min(summary(regfit.fwd)$bic)
model.fwd_bic <- summary(regfit.fwd)$which[fwd.min_bic, -1]

predictors.fwd_adjr2 <- names(which(model.fwd_adjr2 == TRUE))
predictors.fwd_cp <- names(which(model.fwd_cp == TRUE))
predictors.fwd_bic <- names(which(model.fwd_bic == TRUE))

predictors.fwd <- unique(c(predictors.fwd_adjr2, predictors.fwd_cp, predictors.fwd_bic))

# Select best predictors from backward selection
bwd.max_adjr2 <- which.max(summary(regfit.bwd)$adjr2)
model.bwd_adjr2 <- summary(regfit.bwd)$which[bwd.max_adjr2, -1]

bwd.min_cp <- which.min(summary(regfit.bwd)$cp)
model.bwd_cp <- summary(regfit.bwd)$which[bwd.min_cp, -1]

bwd.min_bic <- which.min(summary(regfit.bwd)$bic)
model.bwd_bic <- summary(regfit.bwd)$which[bwd.min_bic, -1]

predictors.bwd_adjr2 <- names(which(model.bwd_adjr2 == TRUE))
predictors.bwd_cp <- names(which(model.bwd_cp == TRUE))
predictors.bwd_bic <- names(which(model.bwd_bic == TRUE))

predictors.bwd <- unique(c(predictors.bwd_adjr2, predictors.bwd_cp, predictors.bwd_bic))

# Combine best predictors from forward and backward
selected_predictors <- unique(c(predictors.fwd, predictors.bwd))

# Regularization: Lasso
train_subset <- uncorrelated_features[, c(selected_predictors)]
train_subset$label <- train_data$Label

# Split train and validation data for lasso
x <- model.matrix(label ~ ., train_subset)[, -1]
y <- as.matrix(train_subset$label)

train_indices <- sample(1:nrow(x), nrow(x) / 2)
validation_indices <- setdiff(1:nrow(x), train_indices)
y_validation <- as.numeric(y[validation_indices])

# Lasso with cross-validation
lasso.cv <- cv.glmnet(x[train_indices, ], y[train_indices], alpha = 1, family = "binomial", type.measure = "auc")
lasso.mod <- glmnet(x[train_indices, ], y[train_indices], lambda = lasso.cv$lambda.min, alpha = 1, family = "binomial")
lasso.pred <- predict(lasso.mod, s = lasso.cv$lambda.min, newx = x[validation_indices, ])
lasso.error <- mean((lasso.pred - y_validation)^2)
lasso.coef <- predict(lasso.mod, s = lasso.cv$lambda.min, type = "coefficients")[1:39, ]
lasso.coef <- lasso.coef[lasso.coef != 0]
lasso.predictors <- names(lasso.coef[-1])
num_lasso_predictors <- length(lasso.predictors)

# Features selected with Lasso
selected_features <- train_subset[, c(lasso.predictors)]
selected_features$label <- make.names(as.factor(train_data$Label))

# Model training and evaluation
sample_size <- floor(0.7 * nrow(selected_features))
indices <- sample(seq_len(nrow(selected_features)), size = sample_size)

train_set <- selected_features[indices, ]
validation_set <- selected_features[-indices, ]

# Seeds
seeds <- vector(mode = "list", length = 11)
for (i in 1:11) seeds[[i]] <- sample.int(1000, 22)

# Train control
ctrl <- trainControl(
  method = "cv",
  number = 10,
  seeds = seeds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Extract feature columns from the validation dataset
features_validation <- validation_set[, names(validation_set) != "label"]

# Model training and evaluation
model_names <- c("glm", "lda", "knn", "LogitBoost", "svmLinear2", "svmLinearWeights")
metrics <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(metrics) <- c("model", "auc", "auc_test", "mcc")

for (model_name in model_names) {
  model <- train(
    label ~ .,
    data = train_set,
    method = model_name,
    metric = "ROC",
    trControl = ctrl
  )
  pred <- predict(model, newdata = features_validation, type = "raw")
  row <- data.frame(
    model_name,
    max(model$result$ROC),
    auc(validation_set$label, as.numeric(pred)),
    mcc(pred, as.factor(validation_set$label))
  )
  names(row) <- c("model", "auc", "auc_test", "mcc")
  metrics <- rbind(metrics, row)
}
metrics

#Test data
test_features <- test_data[, c(lasso.predictors)]

svm.model <- train(
  label ~ .,
  data = train_set,
  method = "svmLinear2",
  metric = "ROC",
  trControl = ctrl
)

test_pred <- predict(svm.model,, newdata =test_features, type = "raw")
test_prob <- predict(svm.model, newdata = test_features, type = "prob")
test_ids <- test_data[1]

# Combine the test IDs, predicted labels, and probabilities into a data frame
results <- data.frame(ID = test_ids, PredictedLabel = as.numeric(test_pred), Probabilities = test_prob[,2])

library(yardstick)

# Read the predictions and true labels
mypredictions <- read.csv('75741_ADCTLres.csv')
mypred <- mypredictions[, 2]
mylabel <- mypredictions[, 4]

# Convert the predictions and labels to factors
mypred <- as.factor(mypred)
mylabel <- as.factor(mylabel)

# Create a confusion matrix
cm <- table(mylabel, mypred)

# Extract the values from the confusion matrix
TN <- cm[1, 1]
FP <- cm[1, 2]
FN <- cm[2, 1]
TP <- cm[2, 2]

# Compute the Matthews Correlation Coefficient (MCC)
mcc <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

accuracy <- (TP + TN) / (TP + FP + FN + TN)

# Print the MCC and accuracy
print(paste("MCC:", mcc))
print(paste("Accuracy:", accuracy))
