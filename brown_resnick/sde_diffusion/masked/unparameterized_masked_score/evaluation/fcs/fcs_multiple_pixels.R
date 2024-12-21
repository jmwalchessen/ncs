library(reticulate)
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

produce_mask <- function(observed_indices, n)
{
  mask <- array(0, dim = c((n**2)))
  mask[observed_indices] <- rep(1, length(observed_indices))
  return(mask)
}

produce_random_mask <- function(m, n)
{
  
  observed_indices <- sample(1:(n**2), m)
  mask <- produce_mask(observed_indices, n)
}

flatten_matrix <- function(twodmatrix, n)
{
  onedarray <- c()
  for(i in 1:n)
  {
    onedarray <- c(onedarray, twodmatrix[i,])
  }
  return(onedarray)
}

brown_resnick_data_generation <- function(number_of_replicates, n, range, smooth)
{
  x <- y <- seq(-10, 10, length = n)
  coord <- expand.grid(x, y)
  y <- SpatialExtremes::rmaxstab(n = number_of_replicates, coord = coord, cov.mod = "brown",
                                 range = range, smooth = smooth)
  return(y)
}

generate_reference_data <- function(number_of_replicates, coord, range, smooth, m, n, mask_file, ref_image_file)
{
  np <- import("numpy")
  ref_image <- brown_resnick_data_generation(number_of_replicates = 1, n = n, range = range, smooth = smooth)
  dim(ref_image) <- c(n,n)
  mask <- produce_random_mask(m,n)
  dim(mask) <- c(n,n)
  np <- import("numpy")
  np$save(mask_file, mask)
  np$save(ref_image_file, ref_image)
}


generate_fcs <- function(mask_file_name, ref_image_name, n, nrep, range, smooth, nugget, fcs_file) 
{
  np <- import("numpy")
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  
  mask <- np$load(mask_file_name)
  ref_image <- exp(np$load(ref_image_name))
  ref_image <- flatten_matrix(ref_image, n)
  mask <- flatten_matrix(mask, n)
  observed_indices <- (1:n**2)[mask == 1]
  observed_spatial_grid <- spatial_grid[observed_indices,]
  observations <- ref_image[observed_indices]
  unobserved_indices <- (1:n**2)[-observed_indices]
  unobserved_observations <- ref_image[unobserved_indices]
  unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
  output <- SpatialExtremes::condrmaxstab(nrep, coord = unobserved_spatial_grid,
                                          cond.coord = observed_spatial_grid,
                                          cond.data = observations,
                                          cov.mod = "brown", 
                                          nugget = nugget, 
                                          range = range,
                                          smooth = smooth)
  condsim <- output$sim
  np$save(fcs_file, condsim)
}

generate_fcs_with_temporary_data <- function(n, nrep, range, smooth, nugget, m)
{
  np <- import("numpy")
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  
  mask <- produce_random_mask(m, n)
  ref_image <- brown_resnick_data_generation(1, n, range, smooth)
  observed_indices <- (1:n**2)[mask == 1]
  observed_spatial_grid <- spatial_grid[observed_indices,]
  observations <- ref_image[observed_indices]
  unobserved_indices <- (1:n**2)[-observed_indices]
  unobserved_observations <- ref_image[unobserved_indices]
  unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
  output <- SpatialExtremes::condrmaxstab(nrep, coord = unobserved_spatial_grid,
                                          cond.coord = observed_spatial_grid,
                                          cond.data = observations,
                                          cov.mod = "brown", 
                                          nugget = nugget, 
                                          range = range,
                                          smooth = smooth)
  return(list(ref_image, mask, output$sim))
}

generate_unconditional_fcs <- function(m, n, nrep, range, smooth, nugget, fcs_file, mask_file, ref_file)
{
  np <- import("numpy")
  conditional_simulations <- array(0, dim = c(nrep, ((n**2)-m)))
  ref_images <- array(0, dim = c(nrep, n**2))
  masks <- array(0, c(nrep, n**2))
  for(i in 1:nrep)
  {
    list_values <- generate_fcs_with_temporary_data(n, 1, range, smooth, nugget, m)
    ref_images[i,] <- list_values[[1]]
    masks[i,] <- list_values[[2]]
    conditional_simulations[i,] <- list_values[[3]]
  }
 np$save(fcs_file, conditional_simulations)
 np$save(ref_file, ref_images)
 np$save(mask_file, masks)
}

generate_unconditional_fcs_multipe_files <- function(n, nrep, range, smooth, nugget, fcs_file, mask_file, ref_file,
                                                     observed_location_numbers)
{
  for(i in 1:length(observed_location_numbers))
  {
    m <- observed_location_numbers[i]
    current_fcs_file <- paste(paste(paste(fcs_file, as.character(m), sep = "_"),
                                                    as.character(nrep), sep = "_"),
                                                    "npy", sep = ".")
    current_mask_file <- paste(paste(paste(mask_file, as.character(m), sep = "_"),
                                                    as.character(nrep), sep = "_"),
                                                    "npy", sep = ".")
    current_ref_file <- paste(paste(paste(ref_file, as.character(m), sep = "_"),
                                                    as.character(nrep), sep = "_"),
                                                    "npy", sep = ".")
    generate_unconditional_fcs(m, n, nrep, range, smooth, nugget, current_fcs_file, current_mask_file, current_ref_file)
  }
}

n <- 32
nrep <- 4000
range <- 3.0
smooth <- 1.5
nugget <- .00001
observed_location_numbers <- seq(1,7,1)
evaluation_folder <- (strsplit(getwd(), "/fcs")[[1]])[1]
extremal_coefficient_and_high_dimensional_folder <- paste(evaluation_folder, "extremal_coefficient_and_high_dimensional_statistics/data/fcs", sep = "/")
fcs_file <- paste(extremal_coefficient_and_high_dimensional_folder,
                  "unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs", sep = "/")
ref_file <- paste(extremal_coefficient_and_high_dimensional_folder,
                  "unconditional_obs_fcs_range_3.0_smooth_1.5_nugget_1e5_obs", sep = "/")
mask_file <- paste(extremal_coefficient_and_high_dimensional_folder,
                  "unconditional_mask_fcs_range_3.0_smooth_1.5_nugget_1e5_obs", sep = "/")
generate_unconditional_fcs_multipe_files(n, nrep, range, smooth, nugget, fcs_file, mask_file, ref_file,
                                 observed_location_numbers)
