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
  return(list(ref_image, mask))
}


time_fcs <- function(n, nrep, range, smooth, nugget, m) 
{
  np <- import("numpy")
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  x <- y <- seq(-10, 10, length = n)
  coord <- expand.grid(x, y)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  reference_data <- generate_reference_data(nrep, coord, range, smooth, m, n)
  ref_image <- reference_data[[1]]
  mask <- reference_data[[2]]
  ref_image <- flatten_matrix(ref_image, n)
  mask <- flatten_matrix(mask, n)
  observed_indices <- (1:n**2)[mask == 1]
  observed_spatial_grid <- spatial_grid[observed_indices,]
  observations <- ref_image[observed_indices]
  unobserved_indices <- (1:n**2)[-observed_indices]
  unobserved_observations <- ref_image[unobserved_indices]
  unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
  time_array <- system.time(output <- SpatialExtremes::condrmaxstab(nrep, coord = unobserved_spatial_grid,
                                                                    cond.coord = observed_spatial_grid,
                                                                    cond.data = observations,
                                                                    cov.mod = "brown", 
                                                                    nugget = nugget, 
                                                                    range = range,
                                                                    smooth = smooth))
  return(time_array)
}

collect_time_fcs <- function(n, nrep, range, smooth, nugget, time_nrep, spatial_location_numbers,
                             user_time_array_file, sys_time_array_file, elapsed_time_array_file)
{
    np <- import("numpy")
    user_time_array <- array(0, dim = c(length(spatial_location_numbers),time_nrep))
    elapsed_time_array <- array(0, dim = c(length(spatial_location_numbers),time_nrep))
    sys_time_array <- array(0, dim = c(length(spatial_location_numbers),time_nrep))
    for(i in 1:length(spatial_location_numbers))
    {
        m <- spatial_location_numbers[i]
        for(j in 1:time_nrep)
        {
            current_time <- time_fcs(n, nrep, range, smooth, nugget, m)
            user_time_array[i,j] <- current_time[1]
            sys_time_array[i,j] <- current_time[2]
            elapsed_time_array[i,j] <- current_time[3]
        }
    }
    np$save(user_time_array_file, user_time_array)
    np$save(sys_time_array_file, sys_time_array)
    np$save(elapsed_time_array_file, elapsed_time_array)
}

#per timing rep for m observed locations, generate mask and reference image and compute single time for 1 cond sim


n <- 32
nrep <- 1
smooth <- 1.5
nugget <- .00001
time_nrep <- 50
spatial_location_numbers <- seq(1,7,1)
range_values <- seq(1,5,1)
for(i in 1:length(range_values))
{
  user_time_array_file <- paste(paste("data/range", as.character(range_values[i]), sep = "_"),
                                    "fcs_user_timing_azure_gpu_1_7_tnrep_50.npy", sep = "_")
  sys_time_array_file <- paste(paste("data/range", as.character(range_values[i]), sep = "_"),
                                    "fcs_system_timing_azure_gpu_1_7_tnrep_50.npy", sep = "_")
  elapsed_time_array_file <- paste(paste("data/range", as.character(range_values[i]), sep = "_"),
                                    "fcs_elapsed_timing_azure_gpu_1_7_tnrep_50.npy", sep = "_")
  collect_time_fcs(n, nrep, range_values[i], smooth, nugget, time_nrep, spatial_location_numbers,
                 user_time_array_file, sys_time_array_file, elapsed_time_array_file)
}