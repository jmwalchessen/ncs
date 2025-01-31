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

generate_reference_data_extreme <- function(number_of_replicates, coord, range, smooth, m, n, mask_file, ref_image_file)
{
  np <- import("numpy")
  extreme_value_cutoff <- 4
  extreme_value <- 0
  mask <- produce_random_mask(m,n)
  while(extreme_value < extreme_value_cutoff)
  {
    ref_image <- brown_resnick_data_generation(number_of_replicates = 1, n = n, range = range, smooth = smooth)
    observed_values <- ref_image[mask == 1]
    print(observed_values)
    extreme_value <- max(log(observed_values))
  }
  dim(ref_image) <- c(n,n)
  dim(mask) <- c(n,n)
  np <- import("numpy")
  np$save(mask_file, mask)
  np$save(ref_image_file, ref_image)
}

generate_reference_data_without_mask <- function(number_of_replicates, coord, range, smooth, n, ref_image_file)
{
  np <- import("numpy")
  ref_image <- brown_resnick_data_generation(number_of_replicates = 1, n = n, range = range, smooth = smooth)
  dim(ref_image) <- c(n,n)
  np <- import("numpy")
  np$save(ref_image_file, ref_image)
}

generate_mask <- function(n, mask_file, m)
{
  np <- import("numpy")
  mask <- produce_random_mask(m, n)
  np$save(mask_file, mask)
}


generate_fcs <- function(mask_file_name, ref_image_name, n, nrep, range, smooth, nugget, fcs_file, m) 
{
  np <- import("numpy")
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  
  mask <- np$load(mask_file_name)
  ref_image <- np$load(ref_image_name)
  ref_image <- flatten_matrix(ref_image, n)
  mask <- flatten_matrix(mask, n)
  observed_indices <- (1:n**2)[mask == 1]
  observed_spatial_grid <- spatial_grid[observed_indices,]
  observations <- ref_image[observed_indices]
  unobserved_indices <- (1:n**2)[-observed_indices]
  unobserved_observations <- ref_image[unobserved_indices]
  unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
  condsim <- array(0, dim = c(nrep,((n**2)-m)))
  for(i in 1:nrep)
  {
    print(i)
    output <- SpatialExtremes::condrmaxstab(1, coord = unobserved_spatial_grid,
                                            cond.coord = observed_spatial_grid,
                                            cond.data = observations,
                                            cov.mod = "brown", 
                                            nugget = nugget, 
                                            range = range,
                                            smooth = smooth,
                                            burnin = 1000,
                                            thin = 100)
    condsim[i,] <- array(output$sim, dim = c(1,((n**2)-m)))
  }
  np$save(fcs_file, condsim)
}

generate_fcs_multiple_ranges <- function(range_values)
{
  n <- 32
  x <- y <- seq(-10, 10, length = n)
  coord <- expand.grid(x, y)
  number_of_replicates <- 10
  smooth <- 1.5
  nugget <- .00001
  ms <- seq(1,7)
  np <- import("numpy")

  for(i in 1:length(range_values))
  {
    ref_folder_name <- paste("data/ranges/ref_image", as.character((range_values[i]-1)), sep = "")
    ref_image_name <- paste(ref_folder_name, "ref_image.npy", sep = "/")
    mask_file_name <- paste(ref_folder_name, "mask.npy", sep = "/")
    generate_reference_data_without_mask(number_of_replicates, coord, range_values[i], smooth, n,
                                           ref_image_name)
  }

  for(j in 1:length(ms))
  {
    for(i in 1:length(range_values))
    {
      np <- import("numpy")
      ref_folder_name <- paste("data/ranges/ref_image", as.character((range_values[i]-1)), sep = "")
      ref_image_name <- paste(ref_folder_name, "ref_image.npy", sep = "/")
      mask_file_name <- paste(ref_folder_name, "mask.npy", sep = "/")
      mask <- produce_random_mask(ms[j], n)
      mask_file_name <- paste(paste(paste(ref_folder_name, "mask_obs", sep = "/"), as.character(ms[j]), sep = "_"), "npy", sep = ".")
      dim(mask) <- c(n,n)
      np$save(mask_file_name, mask)
      fcs_file <- paste(paste(paste(paste(paste("fcs_range", as.character(range_values[i]), sep = "_"),
                                    "smooth_1.5_nugget_1e5_obs", sep = "_"), as.character(ms[j]), sep = "_"),
                                    as.character(number_of_replicates), sep = "_"), "npy", sep = ".")
      fcs_file <- paste(ref_folder_name, fcs_file, sep = "/")
      generate_fcs(mask_file_name, ref_image_name, n, number_of_replicates,
                    range_values[i], smooth, nugget, fcs_file, ms[j])
    }
  }
}

generate_fcs_multiple_ranges_fixed <- function()
{
  n <- 32
  x <- y <- seq(-10, 10, length = n)
  coord <- expand.grid(x, y)
  number_of_replicates <- 4000
  smooth <- 1.5
  nugget <- .00001
  np <- import("numpy")
  ms <- seq(2,7)
  range_values <- seq(1.,5.,1.)


  for(j in 1:length(ms))
  {
    for(i in 1:length(range_values))
    { 
      ref_folder_name <- paste(paste(paste("data/model4/obs", as.character(ms[j]), sep = ""), "ref_image", sep = "/"), as.character((range_values[i]-1)), sep = "")
      ref_image_name <- paste(ref_folder_name, "ref_image.npy", sep = "/")
      mask_file_name <- paste(ref_folder_name, "mask.npy", sep = "/")
      generate_reference_data_extreme(number_of_replicates, coord, range_values[i], smooth, ms[j], n, mask_file_name, ref_image_name)
      fcs_file <- paste(paste(paste(paste(paste("fcs_range", as.character(range_values[i]), sep = "_"),
                                    "smooth_1.5_nugget_1e5_obs", sep = "_"), as.character(ms[j]), sep = "_"),
                                    as.character(number_of_replicates), sep = "_"), "npy", sep = ".")
      fcs_file <- paste(ref_folder_name, fcs_file, sep = "/")
      generate_fcs(mask_file_name, ref_image_name, n, number_of_replicates,
                    range_values[i], smooth, nugget, fcs_file, ms[j])
    }
  }
}


generate_fcs_with_variables <- function(m)
{
  ref_folder <- "data/model4/ref_image"
  ms <- seq(1,7,1)
  n <- 32
  nrep <- 4000
  range <- 3.0
  smooth <- 1.5
  nugget <- .00001
  ref_folder <- paste(ref_folder, as.character((m-1)), sep = "")
  fcs_file <- paste(paste(paste(ref_folder, "fcs_1_by_1_range_3.0_smooth_1.5_nugget_1e5_obs",
                                sep = "/"), as.character(m), sep = "_"), "4000.npy", sep = "_")
  mask_file <- paste(current_ref_folder, "mask.npy", sep = "/")
  ref_image_name <- paste(current_ref_folder, "ref_image.npy", sep = "/")
  generate_fcs(mask_file, ref_image_name, n, nrep, range, smooth, nugget, fcs_file, m)
}




generate_fcs_with_temporary_data <- function(n, nrep, range, smooth, nugget, m)
{
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
                                          smooth = smooth,
                                          burnin = 1000,
                                          thin = 100)
  return(list(ref_image, mask, output$sim))
}

generate_unconditional_fcs <- function(m, n, nrep, range, smooth, nugget, fcs_file, mask_file, ref_file)
{
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
  np <- import("numpy")
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

generate_unconditional_fcs_multiple_files_with_variables <- function(range_values)
{
  n <- 32
  nrep <- 2
  smooth <- 1.5
  nugget <- .00001
  observed_location_numbers <- seq(6,7,1)
  evaluation_folder <- (strsplit(getwd(), "/fcs")[[1]])[1]
  extremal_coefficient_and_high_dimensional_folder <- paste(evaluation_folder, "extremal_coefficient_and_high_dimensional_metrics/data/fcs", sep = "/")
  for(i in 1:length(range_values))
  {
    fcs_file <- paste(paste(paste(extremal_coefficient_and_high_dimensional_folder,
                    "unconditional_fcs_range", sep = "/"), as.character(range_values[i]), sep = "_"), "smooth_1.5_nugget_1e5_obs", sep = "_")
    ref_file <- paste(paste(paste(extremal_coefficient_and_high_dimensional_folder,
                    "unconditional_obs_fcs_range", sep = "/"), as.character(range_values[i]), sep = "_"), "smooth_1.5_nugget_1e5_obs", sep = "_")
    mask_file <- paste(paste(paste(extremal_coefficient_and_high_dimensional_folder,
                    "unconditional_mask_fcs_range", sep = "/"), as.character(range_values[i]), sep = "_"), "smooth_1.5_nugget_1e5_obs", sep = "_")
    generate_unconditional_fcs_multipe_files(n, nrep, range_values[i], smooth, nugget, fcs_file, mask_file, ref_file,
                                  observed_location_numbers)
  } 
}

generate_fcs_with_temporary_data_fixed_mask <- function(n, nrep, range, smooth, nugget, m, mask)
{
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  dim(mask) <- c(n**2)
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
                                          smooth = smooth,
                                          burnin = 1000,
                                          thin = 100)
  return(list(ref_image, output$sim))
}


generate_fixed_locations_unconditional_fcs_multiple_ranges_multipe_obs_with_variables <- function()
{
  range_values <- seq(1.,5.,1)
  ms <- seq(1,7)
  n <- 32
  nrep <- 400
  for(i in 1:len(range_values))
  {
    for(j in 1:len(ms))
    {
      br_images <- array(data = NA, dim = c(rep,n**2))
      condsim <- array(data = NA, dim = c(nrep,((n**2)-ms[j])))
      ref_folder <- paste(paste("data/fcs/unconditional/fixed_locations/obs", as.character(ms[j], sep = ""),
                          "ref_image", sep = "/"), as.character(range_values[i]-1), sep = "")
      mask_file <- paste(ref_folder, "mask.npy", sep = "/")
      generate_mask(n, mask_file, ms[j])
      mask <- np$load(mask_file)
      results <- generate_fcs_with_temporary_data_fixed_mask(n, nrep, range_values[i], smooth, nugget, ms[j], mask)
      br_images[i,j,,] <- results[[1]]
      condsim[i,j,,] <- results[[2]]
      br_images_file <- paste(paste(paste(paste(paste(ref_folder, "true_brown_resnick_images_range", sep = "/"), as.character(range_values[i]),
                                    sep = "_"), "smooth_1.5", sep = "_"), as.character(nrep), sep = "_"), "npy", sep = ".")
      condsim_file <- paste(paste(paste(paste(paste(paste(paste(ref_folder, "unconditional_fcs_fixed_mask_obs", sep = "/"), as.character(ms[j]), sep = ""),
                                                    "range", sep = "_"), as.character(range_values[i]), sep = "_"),
                                                    "smooth_1.5_nugget_1e5", sep = "_"), as.character(nrep), sep = "_"), "npy", sep = ".")

      np$save(br_images_file, br_images)
      np$save(condsim_file, condsim)
    }
  }
} 


generate_fcs_multiple_ranges_fixed()