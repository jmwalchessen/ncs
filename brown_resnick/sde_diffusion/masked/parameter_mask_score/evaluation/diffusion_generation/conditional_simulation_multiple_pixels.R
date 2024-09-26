library(dbscan)
library("parallel")
library("reticulate")
library(SpatialExtremes)
library(R.utils)


produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
}

located_neighboring_pixels <- function(observed_spatial_grid, k, key_location)
{
    knn <- kNN(observed_spatial_grid, k = k, query = key_location)
    id_matrix <- as.matrix(knn$id)
    return(id_matrix)
}

MCMC_interpolation_per_pixel <- function(observed_spatial_grid, observations, k, key_location,
                                            cov_mod, nugget, range, smooth, nrep)
{
    print(key_location)
    id_matrix <- located_neighboring_pixels(observed_spatial_grid, k, key_location)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    print(cond_coord)
    output <- SpatialExtremes::condrmaxstab(nrep, coord = key_location,
              cond.coord = cond_coord,
              cond.data = cond_data,
              cov.mod = cov_mod, 
              nugget = nugget, 
              range = range,
              smooth = smooth)
    condsim <- output$sim
}

interruptor <- function(FUN,args, time.limit, ALTFUN){

  results <- 
    tryCatch({
      withTimeout({FUN(args)}, timeout=time.limit)
    }, error = function(e){
      if(grepl("reached elapsed time limit",e$message))
        ALTFUN(args) else
          paste(e$message,"EXTRACTERROR")
      })

  #if(grepl("EXTRACTERROR",results)){
    #print(gsub("EXTRACTERROR","",results))
    #results <- array(1,args$nrep)
  #} 

  return(results)
} 

alternative_MCMC_interpolation_per_pixel_via_mask <- function(argsList)
{
    return(array(NA, dim = c(1, argsList$nrep)))
}

produce_mcmc_interpolation_per_pixel_via_mask <- function(argsList)
{
    n <- argsList$n
    range <- argsList$range
    smooth <- argsList$smooth
    nugget <- argsList$nugget
    cov_mod <- argsList$cov_mod
    mask_file_name <- argsList$mask_file_name
    ref_image_name <- argsList$ref_image_name
    neighbors <- argsList$neighbors
    nrep <- argsList$nrep
    missing_index <- argsList$missing_index

    np <- import("numpy")
    s1 <- s2 <- seq(-10, 10, length.out = n)
    s <- cbind(s1, s2)
    spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)

    mask <- np$load(mask_file_name)
    observations <- exp(np$load(ref_image_name))
    dim(observations) <- c(n**2)
    dim(mask) <- c(n**2)
    observed_indices <- (1:n**2)[mask == 1]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    observations <- observations[observed_indices]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_observations <- observations[unobserved_indices]
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    key_location <- unobserved_spatial_grid[missing_index,]
    condsim <- MCMC_interpolation_per_pixel(observed_spatial_grid, observations, neighbors, key_location,
                                            cov_mod, nugget, range, smooth, nrep)
    return(condsim)
}


produce_mcmc_interpolation_per_pixel_via_mask_interrupted <- function(n, range, smooth, nugget, cov_mod, mask_file_name, ref_image_name,
                                                          neighbors, nrep, missing_index)
{
    x <- interruptor(FUN = produce_mcmc_interpolation_per_pixel_via_mask, args = list(n = n, range = range, smooth = smooth,
                                                                                      nugget = nugget, cov_mod = cov_mod,
                                                                                      mask_file_name = mask_file_name,
                                                                                      ref_image_name = ref_image_name,
                                                                                      neighbors = neighbors, nrep = nrep,
                                                                                      missing_index = missing_index),
                                                                                      time.limit = 60, ALTFUN = alternative_MCMC_interpolation_per_pixel_via_mask)
    return(x)
}




produce_local_conditional_simulation_for_multiple_pixels <- function(indices, n, range, smooth, nugget, cov_mod, mask_file_name,
                                                                     ref_image_name, neighbors, nrep, condsim_file_name)
{
  for(missing_index in indices)
  {
    y <- produce_mcmc_interpolation_per_pixel_via_mask_interrupted(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                                   ref_image_name, neighbors, nrep, missing_index)
    current_condsim_file <- paste(paste(condsim_file_name, as.character(missing_index), sep = "_"), "npy", sep = ".")
    np <- import("numpy")
    np$save(current_condsim_file, y)
  }
}

produce_local_conditional_simulation_multiple_references <- function(indices, n, range_values, smooth, nugget, cov_mod,
                                                                     neighbors, nrep, condsim_file_name, model_folder,
                                                                     ref_image_indices)
{
  for(i in length(ref_image_indices))
  {
    ref_image_folder <- paste(paste(model_folder, "ref_image", sep = "/"), as.character(ref_image_indices[i]), sep = "")
    ref_image_name <- paste(ref_image_folder, "ref_image.npy", sep = "/")
    mask_file_name <- paste(ref_image_folder, "mask.npy", sep = "/")
    current_condsim_file <- paste(ref_image_folder, condsim_file_name, sep = "/")
    produce_local_conditional_simulation_for_multiple_pixels(indices, n, range_values[i], smooth, nugget, cov_mod, mask_file_name,
                                                                     ref_image_name, neighbors, nrep, current_condsim_file_name)
  }
}

indices <- [10*i for i in range(0, 500)]
n <- 32
range_values <- [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
smooth <- 1.6
nugget <- .0001
cov_mod <- "brown"
neighbors <- 5
nrep <- 4000
condsim_file_name <- paste(paste("local_conditional_simulation/local_conditional_simulation_neighbors", as.character(neighbors), sep = "_"),
                                  as.character(nrep), sep = "_")
condsim_file_name <- paste(condsim_file_name, sep = "/")
model_folder <- "data/model2"
ref_image_indices <- [1,2,3,4,5,6]
produce_local_conditional_simulation_multiple_references(indices, n, range_values, smooth, nugget, cov_mod,
                                                         neighbors, nrep, condsim_file_name, model_folder,
                                                         ref_image_indices)