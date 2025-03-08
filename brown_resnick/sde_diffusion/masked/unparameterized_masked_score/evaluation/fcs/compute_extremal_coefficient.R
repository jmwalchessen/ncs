library(SpatialExtremes)
library(reticulate)
library(parallel)

np <- import("numpy")


create_folder <- function(range, obs)
{
    ref_folder <- paste(paste(paste("data/unconditional/fixed_locations/obs", as.character(obs), sep = ""), "ref_image", as.character(range-1), sep = ""))
    return(ref_folder)
}

#scale, sep = "/")location, and shape parameters for BR are 1,1,1
compute_true_extremal_coefficient <- function(number_of_replicates,
                                              range, smooth, nbins, ext_file, n, obs)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  ref_folder <- create_folder(range,obs)
  y <- np$load(paste(paste(paste(ref_folder, "true_brown_resnick_images_range", sep = "/"), as.character(range), sep = "_"), ".0_smooth_1.5_4000.npy", sep = ""))
  dim(y) <- c(number_of_replicates, n.size)
  mado <- madogram(data = y, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ext_file, mado)
  dim(y) <- c(number_of_replicates,n,n)
}

compute_true_extremal_coefficient_with_variables <- function()
{
  number_of_replicates <- 4000
  smooth <- 1.5
  nbins <- 100
  n <- 32
  obs_numbers <- seq(1,7,1)
  range_values <- seq(1.,5.,1.)
  for(i in 1:length(range_values))
  {
    for(j in 1:length(obs_numbers))
    {
        ref_folder <- create_folder(range_values[i], obs_numbers[j])
        ext_file <- paste(paste(paste(ref_folder, "true_extremal_coefficient_range", sep = "/"), as.character(range_values[i]), sep = "_"),
                            ".0_smooth_1.5_nbins_100_4000.npy", sep= "")
        compute_true_extremal_coefficient(number_of_replicates, range_values[i], smooth, nbins, ext_file, n, obs_numbers[j])
    }
  }
}


compute_ncs_extremal_coefficient <- function(number_of_replicates, range, smooth, nbins, ncs_file, ncs_ext_file, n)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  ncs_images <- exp(np$load(ncs_file))
  print(dim(ncs_images))
  dim(ncs_images) <- c(number_of_replicates,(n**2))
  mado <- madogram(data = ncs_images, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(ncs_ext_file, mado)
}

compute_ncs_extremal_coefficient_with_variables <- function()
{
  number_of_replicates <- 4000
  range_values <- seq(1.,5.,1.)
  obs_numbers <- seq(1,7,1)
  number_of_replicates_per_call <- 50
  smooth <- 1.5
  nbins <- 100
  n <- 32
  for(i in 1:length(range_values))
  {
    for(j in 1:length(obs_numbers))
    {
        ref_folder <- create_folder(range_values[i], obs_numbers[j])
        ncs_file <- paste(paste(paste(paste(paste(ref_folder, "diffusion/unconditional_fixed_ncs_images_range", sep = "/"),
                            as.character(range_values[i]), sep = "_"), ".0_smooth_1.5", sep = ""),
                            as.character(nrep), sep = "_"), "npy", sep = ".")
        ncs_ext_file <- paste(paste(paste(paste(ref_folder, "brown_resnick_ncs_extremal_matrix_bins_100_obs", sep = "/"),
                        as.character(obs_numbers[j]), sep = ""), "range_", sep = "_"), "smooth_1.5_4000.npy", sep = "_")
        compute_ncs_extremal_coefficient(number_of_replicates, range, smooth, nbins, ncs_file, ncs_ext_file, n)
    }
    }
}

compute_fcs_extremal_coefficient <- function(number_of_replicates, range, smooth, nbins, fcs_file, fcs_ext_file, n, m)
{
  n.size <- 1024
  nn <- sqrt(n.size)
  x <- y <- seq(-10, 10, length = nn)
  coord <- expand.grid(x, y)
  fcs_images <- np$load(fcs_file)
  print(dim(fcs_images))
  dim(fcs_images) <- c(number_of_replicates,(n**2))
  mado <- madogram(data = fcs_images, coord = as.matrix(coord), which = "ext", n.bins = nbins)
  np$save(fcs_ext_file, mado)
}

compute_fcs_extremal_coefficient_with_variables <- function()
{
  number_of_replicates <- 4000
  ranges <- seq(1.,5.,1.)
  smooth <- 1.5
  nbins <- 100
  ms <- seq(1,7,1)
  n <- 32
  for(i in 1:length(ms))
  { 
    for(j in 1:length(ranges))
    {
        ref_folder <- create_folder(ranges[j], ms[i])
        fcs_file <- paste(paste(paste(paste(ref_folder, "processed_unconditional_fcs_fixed_mask_range", as.character(ranges[j]), sep = "_"),
                            ".0_smooth_1.5_nugget_1e5_obs", sep = ""),
                            as.character(m), sep = "_"), "4000.npy", sep = "_")
        fcs_ext_file <- paste(paste(paste(paste(paste(ref_folder, "extremal_coefficient_fcs_range", sep = "/"), as.character(ranges[j]), sep = "_"),
                                ".0_smooth_1.5_nugget_1e5_obs", sep = ""), as.character(m),
                                sep = "_"), "4000.npy", sep = "_")
        compute_fcs_extremal_coefficient(number_of_replicates, ranges[j], smooth, nbins, fcs_file, fcs_ext_file, n, m)
    }
  }
}

compute_fcs_extremal_coefficient_with_variables()
compute_ncs_extremal_coefficient_with_variables()
compute_true_extremal_coefficient_with_variables()