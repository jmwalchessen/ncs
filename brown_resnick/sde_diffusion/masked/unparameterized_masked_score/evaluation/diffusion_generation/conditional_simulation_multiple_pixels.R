library(dbscan)
library("parallel")
library("reticulate")
library(SpatialExtremes)
library(R.utils)

args = commandArgs(trailingOnly=TRUE)
range <- as.numeric(args[1])
smooth <- as.numeric(args[2])
nugget <- as.numeric(args[3])
ref_image_name <- as.character(args[4])
mask_file_name <- as.character(args[5])
condsim_file_name <- as.character(args[6])
cov_mod <- as.character(args[7])
neighbors <- as.numeric(args[8])
n <- as.numeric(args[9])
nrep <- as.numeric(args[10])
missing_index_start <- as.numeric(args[11])
missing_index_end <- as.numeric(args[12])

produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    return(mask)
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

located_neighboring_pixels <- function(observed_spatial_grid, k, key_location)
{
    knn <- kNN(observed_spatial_grid, k = k, query = key_location)
    id_matrix <- as.matrix(knn$id)
    return(id_matrix)
}

MCMC_interpolation_per_pixel <- function(observed_spatial_grid, observations, k, key_location,
                                            cov_mod, nugget, range, smooth, nrep)
{
    print("key location")
    print(key_location)
    id_matrix <- located_neighboring_pixels(observed_spatial_grid, k, key_location)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    print("neighbor obs")
    print(log(cond_data))
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
    ref_image <- exp(np$load(ref_image_name))
    ref_image <- flatten_matrix(ref_image, n)
    print(ref_image[1:32])
    mask <- flatten_matrix(mask, n)
    print(mask[1:32])
    observed_indices <- (1:n**2)[mask == 1]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    observations <- ref_image[observed_indices]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_observations <- ref_image[unobserved_indices]
    print("key obs")
    print(log(unobserved_observations[missing_index]))
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




for(missing_index in missing_index_start:missing_index_end)
{
    print(missing_index)
    y <- produce_mcmc_interpolation_per_pixel_via_mask_interrupted(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                                   ref_image_name, neighbors, nrep, missing_index)
    current_condsim_file <- paste(paste(condsim_file_name, as.character(missing_index), sep = "_"), "npy", sep = ".")
    np <- import("numpy")
    np$save(current_condsim_file, y)
}
rm(list = ls())