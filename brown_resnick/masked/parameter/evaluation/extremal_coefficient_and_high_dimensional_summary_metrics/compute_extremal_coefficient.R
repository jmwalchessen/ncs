library(SpatialExtremes)
library(reticulate)
library(parallel)
source("brown_resnick_data_generation.R")

np <- import("numpy")




#scale, location, and shape parameters for BR are 1,1,1
compute_true_extremal_coefficient_via_simulation <- function(number_of_replicates, number_of_replicates_per_call,
                                                             range, smooth, nbins, ext_file, image_file)
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
  dim(y) <- c(number_of_replicates, nn**2)
  np$save(image_file, y)
}

compute_true_extremal_coefficient_multiple_ranges <- function(number_of_replicates, number_of_replicates_per_call,
                                                              ranges, smooth, nbins, ext_file, image_file)
{
  for(i in 1:length(ranges))
  {
    current_ext_file <- paste(paste(paste(paste(paste(paste(paste(ext_file, "smooth", sep = "_"), as.character(smooth),
                                               sep = "_"), "range", sep = "_"), as.character(ranges[i]), sep = "_"),
                                               "nbins", sep = "_"), as.character(nbins), sep = "_"), "npy", sep = ".")
    current_image_file <- paste(paste(image_file, as.character(ranges[i]), sep = "_"), "npy", sep = ".")
    compute_true_extremal_coefficient_via_simulation(number_of_replicates, number_of_replicates_per_call, ranges[i], smooth, nbins, current_ext_file, current_image_file)
  }
}

compute_ncs_extremal_coefficient <- function(number_of_replicates, range, smooth, nbins, ncs_file, ext_file, n)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  ncs_images <- exp(np$load(ncs_file))
  dim(ncs_images) <- c(number_of_replicates,(n**2))
  mado <- madogram(data = ncs_images, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
}


compute_true_extremal_coefficient_multiple_ranges_fixed_values <- function()
{
  number_of_replicates <- 4000
  number_of_replicates_per_call <- 50
  ranges <- seq(1.,5.,1.)
  smooth <- 1.5
  nbins <- 500
  ext_file <- "data/true/extremal_coefficient"
  image_file <- "data/true/brown_resnick_images_random05_smooth_1.5_range"
  compute_true_extremal_coefficient_multiple_ranges(number_of_replicates, number_of_replicates_per_call,
                                                    ranges, smooth, nbins, ext_file, image_file)
}


number_of_replicates <- 4000
range <- 5.
smooth <- 1.5
nbins <- 500
ncs_file <- "data/ncs/model4/brown_resnick_ncs_images_range_5.0_smooth_1.5_4000.npy"
ext_file <- "data/ncs/model4/ncs_extremal_coefficient_smooth_1.5_range_5_nbins_500.npy"
n <- 32
compute_ncs_extremal_coefficient(number_of_replicates, range, smooth, nbins, ncs_file, ext_file, n)



