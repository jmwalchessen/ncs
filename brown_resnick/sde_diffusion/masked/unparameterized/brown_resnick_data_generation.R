library(SpatialExtremes)
library(parallel)
library(reticulate)

args = commandArgs(trailingOnly=TRUE)
range <- as.numeric(args[1])
smooth <- as.numeric(args[2])
number_of_replicates <- as.numeric(args[3])
seed <- as.numeric(args[4])

n.size <- 961
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)
number_of_replicates_per_call <- 50
calls <- as.integer(number_of_replicates/number_of_replicates_per_call)
repnumberslist <- rep(number_of_replicates_per_call, calls)
repnumberslist <- append(repnumberslist, (number_of_replicates %% number_of_replicates_per_call))

simulate_data_per_core <- function(number_of_reps, nn, coord, range, smooth)
{
    y <- rmaxstab(n = number_of_reps, coord = coord, cov.mod = "brown", range = range, smooth = smooth)
    return(y)
}

collect_data <- function(parallel_output, nn, number_of_replicates_per_call)
{
    m <- length(parallel_output)
    y <- array(0, dim = c(number_of_replicates_per_call*(m-1), nn, nn))
    for (i in 1:(m-1))
    {
        y[((i-1)*number_of_replicates_per_call+1):(i*number_of_replicates_per_call),,] <- parallel_output[[i]]
    }
    return(y)
}


cores <- (detectCores(logical = TRUE))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))
clusterExport(cluster, c("nn", "coord", "range", "smooth", "simulate_data_per_core",
                         "repnumberslist", "rmaxstab"))

y <- parSapply(cluster, repnumberslist, function(repsnumber)
{simulate_data_per_core(repsnumber, nn, coord, range, smooth)})
stopCluster(cluster)

np <- import("numpy")
y <- collect_data(y, nn, number_of_replicates_per_call)
np$save("temporary_brown_resnick_samples.npy", y)
rm(list = ls())