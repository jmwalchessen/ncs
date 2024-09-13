library(parallel)
library(reticulate)
library(devtools)
library(SpatialExtremes)

args = commandArgs(trailingOnly=TRUE)
range <- as.numeric(args[1])
smooth <- as.numeric(args[2])
seed <- as.numeric(args[4])

n.size <- 625
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)
s <- cbind(x,y)
number_of_replicates <- 1

simulate_data_per_core <- function(number_of_replicates, nn, s, range, smooth)
{
    y <- SpatialExtremes::rmaxstab(n = number_of_replicates, coord = s, cov.mod = "brown", range = range, smooth = smooth, grid = TRUE)
    return(y)
}

collect_data <- function(parallel_output, nn)
{
    m <- length(parallel_output)
    y <- array(0, dim = c(1, (nn**2)))
    for (i in 1:m)
    {
        y[((i-1)*+1):i,] <- parallel_output[[i]]
    }
    return(y)
}

repnumberslist <- list(number_of_replicates)
cores <- (detectCores(logical = TRUE))
y <- mclapply(repnumberslist, function(repsnumber)
simulate_data_per_core(repsnumber, nn, s, range, smooth), mc.cores = cores)
y <- collect_data(y, nn)
np <- import("numpy")
np$save("temporary_brown_resnick_samples.npy", y)
rm(list = ls())