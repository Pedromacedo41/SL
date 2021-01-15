serialize_tree <- function(elem){
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


options(java.parameters = "-Xmx5g")
library(bartMachine)
set_bart_machine_num_cores(6)

data(automobile)
automobile = na.omit(automobile)

y <- automobile$log_price
X <- automobile; X$log_price <- NULL

bart_machine <- bartMachine(X,y)
raw_data <- extract_raw_node_data(bart_machine, g= 250)

texts = serialize_all_trees(raw_data)
write.table(texts, "mydata.txt", quote=FALSE, sep="", col.names = FALSE, row.names = FALSE)



