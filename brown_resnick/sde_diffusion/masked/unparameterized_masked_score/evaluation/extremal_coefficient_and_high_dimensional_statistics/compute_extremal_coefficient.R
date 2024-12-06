library(SpatialExtremes)
library(reticulate)
library(parallel)
source("brown_resnick_data_generation.R")

np <- import("numpy")


#scale, location, and shape parameters for BR are 1,1,1
compute_true_extremal_coefficient_simulate_data <- function(number_of_replicates, number_of_replicates_per_call,
                                              range, smooth, nbins, ext_file)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  calls <- as.integer(number_of_replicates/number_of_replicates_per_call)
  repnumberslist <- rep(number_of_replicates_per_call, calls)
  y <- simulate_data_across_cores(repnumberslist, nn, coord, range, smooth, number_of_replicates_per_call)
  mado <- madogram(data = y, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
  dim(y) <- c(number_of_replicates, n, n)
  np$save(true_file, y)
}

compute_true_extremal_coefficient <- function(true_file, number_of_replicates, n,
                                              range, smooth, nbins, ext_file)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  y <- np$load(true_file)
  dim(y) <- c(number_of_replicates, (n**2))
  mado <- madogram(data = y, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
}



compute_ncs_extremal_coefficient <- function(number_of_replicates, range, smooth, nbins, ncs_file, ext_file, n)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  ncs_images <- exp(np$load(ncs_file))
  dim(ncs_images) <- c(number_of_replicates,n**2)
  mado <- madogram(data = ncs_images, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
}


number_of_replicates <- 4000
number_of_replicates_per_call <- 50
calls <- 80
range <- 3.0
smooth <- 1.5
nbins <- 100
n <- 32
true_file <- "data/true/brown_resnick_images_range_3.0_smooth_1.5_4000.npy"
ncs_file <- "data/ncs/model4/brown_resnick_ncs_images_range_3.0_smooth_1.5_4000_random0.5.npy"
ncs_ext_file <- "data/ncs/model4/ncs_extremal_coefficient_range_3.0_smooth_1.5_bins_100_4000_random0.5.npy"
true_ext_file <- "data/true/extremal_coefficient_range_3.0_smooth_1.5_bins_100_4000_random0.5.npy"
compute_true_extremal_coefficient(true_file, number_of_replicates, n,
                                  range, smooth, nbins, true_ext_file)
compute_ncs_extremal_coefficient(number_of_replicates, range, smooth, nbins, ncs_file, ncs_ext_file, n)

