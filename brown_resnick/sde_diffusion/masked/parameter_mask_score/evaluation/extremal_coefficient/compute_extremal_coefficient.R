library(SpatialExtremes)
library(reticulate)
library(parallel)

np <- import("numpy")

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

simulate_data_across_cores <- function(repnumberslist, nn, coord, range, smooth)
{
  cores <- (detectCores(logical = TRUE))
  y <- mclapply(repnumberslist, function(nrep)
                simulate_data_per_core(nrep, nn, coord, range, smooth), mc.cores = cores)
  y <- collect_data(y, nn, number_of_replicates_per_call)
  return(y)
}

#scale, location, and shape parameters for BR are 1,1,1
compute_true_extremal_coefficient <- function(number_of_replicates, number_of_replicates_per_call,
                                              range, smooth, nbins, ext_file)
{
  print(number_of_replicates)
  print(number_of_replicates_per_call)
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  calls <- as.integer(number_of_replicates/number_of_replicates_per_call)
  repnumberslist <- rep(number_of_replicates_per_call, calls)
  print(repnumberslist)
  y <- simulate_data_across_cores(repnumberslist, nn, coord, range, smooth)
  mado <- madogram(data = y, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
}

compute_true_extremal_coefficient_multiple_ranges <- function(number_of_replicates, number_of_replicates_per_call,
                                                              ranges, smooth, nbins, ext_file)
{
  for(i in 1:length(ranges))
  {
    current_ext_file <- paste(paste(paste(paste(paste(paste(paste(ext_file, "smooth", sep = "_"), as.character(smooth),
                                               sep = "_"), "range", sep = "_"), as.character(ranges[i]), sep = "_"),
                                               "nbins", sep = "_"), as.character(nbins), sep = "_"), "npy", sep = ".")
    compute_true_extremal_coefficient(number_of_replicates, number_of_replicates_per_call, ranges[i], smooth, nbins, current_ext_file)
  }
}

compute_ncs_extremal_coefficient <- function(number_of_replicates, range, smooth, nbins, ncs_file, ext_file)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  ncs_images <- np$load(ncs_file)
  dim(ncs_images) <- c(number_of_replicates,1)
  mado <- madogram(data = ncs_images, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
}

compute_ncs_extremal_coefficient_multiple_ranges <- function(number_of_replicates, ranges, smooth, p, nbins, 
                                                             ref_folder, model_name, ext_file)
{
  for(i in 1:length(ranges))
  {
    current_ncs_file <- 
    current_ext_file <- paste(paste(paste(paste(paste(paste(paste(ext_file, "smooth", sep = "_"), as.character(smooth),
                                                            sep = "_"), "range", sep = "_"), as.character(ranges[i]), sep = "_"),
                                          "nbins", sep = "_"), as.character(nbins), sep = "_"), "npy", sep = ".")
    current_ncs_file <- paste(paste(paste(paste(paste(paste(paste(paste(ref_folder, model_name, sep = "/"), "range", sep = "_"),
                              as.character(ranges[i]), sep = "_"), "smooth", sep = "_"), "random", sep = "_"),
                              as.character(p), sep = ""), as.character(number_of_replicates), sep = "_"), "npy",
                              sep = ".")
    compute_ncs_extremal_coefficient(number_of_replicates, ranges[i], smooth, nbins, current_ncs_file,
                                     current_ext_file)
  }
}


compute_true_extremal_coefficient_multiple_ranges_fixed_values <- function()
{
  number_of_replicates <- 4000
  number_of_replicates_per_call <- 50
  ranges <- seq(1.,5.,1.)
  smooth <- 1.5
  nbins <- 100
  ext_file <- "data/true/extremal_coefficient"
  compute_true_extremal_coefficient_multiple_ranges(number_of_replicates, number_of_replicates_per_call,
                                                    ranges, smooth, nbins, ext_file)
}


compute_ncs_extremal_coefficient_multiple_ranges_fixed_values <- function()
{
  number_of_replicates <- 4000
  ranges <- seq(1.,5.,1.)
  smooth <- 1.5
  nbins <- 100
  ext_file <- "data/ncs/model4/extremal_coefficient"
  compute_ncs_extremal_coefficient_multiple_ranges(number_of_replicates, ranges, smooth, nbins, ext_file)
}



