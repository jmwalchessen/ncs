library(parallel)
library(reticulate)
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

args = commandArgs(trailingOnly=TRUE)
range <- as.numeric(args[1])
smooth <- as.numeric(args[2])
number_of_replicates <- as.numeric(args[3])
seed <- as.numeric(args[4])

n.size <- 1024
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)
number_of_replicates_per_call <- 10
calls <- as.integer(number_of_replicates/number_of_replicates_per_call)
repnumberslist <- rep(number_of_replicates_per_call, calls)

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

cores <- (detectCores(logical = TRUE))
y <- mclapply(repnumberslist, function(repsnumber)
simulate_data_per_core(repsnumber, nn, coord, range, smooth), mc.cores = cores)
y <- collect_data(y, nn, number_of_replicates_per_call)
np <- import("numpy")
np$save("temporary_brown_resnick_samples.npy", y)
rm(list = ls())