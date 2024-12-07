library(SpatialExtremes)
library(reticulate)
library(parallel)

simulate_data_per_core <- function(number_of_replicates, nn, coord, range, smooth)
{
  y <- SpatialExtremes::rmaxstab(n = number_of_replicates, coord = coord, cov.mod = "brown", range = range, smooth = smooth)
  return(y)
}

collect_data <- function(parallel_output, nn, number_of_replicates_per_call)
{
  m <- length(parallel_output)
  y <- array(0, dim = c(number_of_replicates_per_call*m, (nn**2)))
  for (i in 1:m)
  {
    y[((i-1)*number_of_replicates_per_call+1):(i*number_of_replicates_per_call),] <- parallel_output[[i]]
  }
  return(y)
}

simulate_data_across_cores <- function(repnumberslist, nn, coord, range, smooth, number_of_replicates_per_call)
{
  cores <- (detectCores(logical = TRUE))
  y <- mclapply(repnumberslist, function(nrep)
    simulate_data_per_core(nrep, nn, coord, range, smooth), mc.cores = cores)
  y <- collect_data(y, nn, number_of_replicates_per_call)
  return(y)
}