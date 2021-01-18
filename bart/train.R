serialize_tree <- function(elem) {
  str <- ""
  if(is.na(elem$n_eta)){
    str <- paste(str,toString(0), sep="")
  }else{
    str <- paste(str,elem$n_eta, sep="")
  }
  if(is.na(elem$splitAttributeM)){
    str <- paste(str,toString(0), sep="@")
  }else{
    str <- paste(str,elem$splitAttributeM, sep="@")
  }
  if(is.na(elem$splitValue)){
    str <- paste(str,toString(0), sep="@")
  }else{
    str <- paste(str,elem$splitValue, sep="@")
  }
  if(is.na(elem$y_pred)){
    str <- paste(str,toString(0), sep="@")
  }else{
    str <- paste(str,elem$y_pred, sep="@")
  }
  if(is.na(elem$y_avg)){
    str <- paste(str,toString(0), sep="@")
  }else{
    str <- paste(str,elem$y_avg, sep="@")
  }
  if(is.na(elem$posterior_var)){
    str <- paste(str,toString(0), sep="@")
  }else{
    str <- paste(str,elem$posterior_var, sep="@")
  }
  if(is.na(elem$posterior_mean)){
    str <- paste(str,toString(0), sep="@")
  }else{
    str <- paste(str,elem$posterior_mean, sep="@")
  }
  if(!is.na(elem$left)){
    str = paste(str, serialize_tree(elem$left))
  }else{
    str = paste(str, "#")
  }
  if(!is.na(elem$right)){
    str = paste(str, serialize_tree(elem$right))
  }else{
    str = paste(str, "#")
  }
  return(str)
}

serialize_all_trees <- function(raw_data){
  c = c()
  for(i in 1:length(raw_data)){
    c <- c(c, serialize_tree(raw_data[i][[1]]))
  }
  return(c)
}

computeRMSE <- function(bart_machine, X, y) {
    y_hat = predict(bart_machine, X)
    return(sqrt(mean((y_hat - y) ** 2)))
}

# Read command line arguments
args = commandArgs(trailingOnly=TRUE)

if (length(args) != 2) {
    stop("This script takes two arguments: dataset name and run number.")
}

# Set seed
run = as.integer(args[2])
set.seed(run)

X <- read.csv(paste("data/X_", args[1], ".csv", sep=""), header=FALSE)
y <- read.csv(paste("data/y_", args[1], ".csv", sep=""), header=FALSE)

y <- y$V1

# 50/25/25 split into train/validation/test
n <- nrow(X)
trainEnd <- floor(0.5 * n)
valEnd <- floor(0.75 * n)

X_train = X[1:trainEnd, ]
X_val = X[trainEnd:valEnd, ]
X_test = X[valEnd:n, ]

y_train = y[1:trainEnd]
y_val = y[trainEnd:valEnd]
y_test = y[valEnd:n]

# Train BART model
options(java.parameters = "-Xmx5g")
library(bartMachine)
set_bart_machine_num_cores(6)

bart_machine <- bartMachine(X_train, y_train,
    num_trees=30,
    num_burn_in=100,
    num_iterations_after_burn_in = 1000
)

# Compute validation and test RMSE
train_rmse = computeRMSE(bart_machine, X_train, y_train)
val_rmse = computeRMSE(bart_machine, X_val, y_val)
test_rmse = computeRMSE(bart_machine, X_test, y_test)
print(paste("Train RMSE:", train_rmse))
print(paste("Validation RMSE:", val_rmse))
print(paste("Test RMSE:", test_rmse))

# Export raw data
raw_data <- extract_raw_node_data(bart_machine, g=250)
texts <- serialize_all_trees(raw_data)
write.table(texts, paste("models/bart_", args[1], args[2], ".model", sep=""),
    quote=FALSE, sep="", col.names=FALSE, row.names=FALSE)

# Record train/validation/test scores
scores <- data.frame(train_rmse, val_rmse, test_rmse)
write.csv(scores, paste("logs/bart_", args[1], args[2], ".csv", sep=""),
  quote=FALSE, row.names=FALSE)