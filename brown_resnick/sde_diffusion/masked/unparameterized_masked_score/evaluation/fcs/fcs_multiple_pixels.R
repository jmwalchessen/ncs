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

generate_fcs_with_temporary_data <- function(n, nrep, range, smooth, nugget)
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
  return(condsim)
}

generate_unconditional_fcs_multipe_files <- function(ref_numbers, m, n, nrep, range, smooth, nugget, fcs_file)
{
  for(i in 1:length(ref_numbers))
  {
    ref_folder <- paste("data/model4/ref_image", as.character(ref_numbers[i]), sep = "")
    mask_file_name <- paste(ref_folder, "mask.npy", sep = "/")
    ref_image_name <- paste(ref_folder, "ref_image.npy", sep = "/")
    fcs_file_name <- paste(paste(paste(fcs_file, as.character(m), sep = "_"), as.character(nrep), sep = "_"), "npy", sep = ".")
    fcs_file_name <- paste(ref_folder, fcs_file_name, sep = "/")
    generate_fcs(mask_file_name, ref_image_name, n, nrep, range, smooth, nugget, fcs_file_name)
  }
}

generate_unconditional_fcs <- function(m, n, nrep, range, smooth, nugget, fcs_file)
{
  np <- import("numpy")
  conditional_simulations <- array(0, dim = c(nrep, ((n**2)-m)))
  for(i in 1:nrep)
  {
    print(i)
    conditional_simulations[i,] <- generate_fcs_with_temporary_data(n, 1, range, smooth, nugget)
  }
 np$save(fcs_file, conditional_simulations) 
}

generate_unconditional_fcs_multipe_files <- function(n, nrep, range, smooth, nugget, fcs_file,
                                             observed_location_numbers)
{
  for(i in 1:length(observed_location_numbers))
  {
    m <- observed_location_numbers[i]
    current_fcs_file <- paste(paste(paste(fcs_file, as.character(m), sep = "_"),
                                                    as.character(nrep), sep = "_"),
                                                    "npy", sep = ".")
    generate_unconditional_fcs(m, n, nrep, range, smooth, nugget, current_fcs_file)
  }
}

n <- 32
nrep <- 1
range <- 3.0
smooth <- 1.5
nugget <- .00001
observed_location_numbers <- seq(5,6,1)
evaluation_folder <- (strsplit(getwd(), "/lcs")[[1]])[1]
extremal_coefficient_and_high_dimensional_folder <- paste(evaluation_folder, "extremal_coefficient_and_high_dimensional_statistics/data/fcs", sep = "/")
fcs_file <- paste(extremal_coefficient_and_high_dimensional_folder,
                  "unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs", sep = "/")
generate_unconditional_fcs_multipe_files(n, nrep, range, smooth, nugget, fcs_file,
                                 observed_location_numbers)
