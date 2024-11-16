library(parallel)
library(reticulate)
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)



n.size <- 1024
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)


simulate_data_per_core <- function(number_of_replicates, nn, coord, range, smooth)
{
  print(number_of_replicates)
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

simulate_data_across_cores <- function(number_of_replicates_per_call, calls, nn, coord, range, smooth)
{
  cores <- (detectCores(logical = TRUE))
  repnumberslist <- rep(number_of_replicates_per_call, calls)
  y <- mclapply(repnumberslist, function(repsnumber) simulate_data_per_core(repsnumber, nn, coord, range, smooth), mc.cores = cores)
  y <- collect_data(y, nn, number_of_replicates_per_call)
}

