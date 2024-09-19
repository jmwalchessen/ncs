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
missing_index_start1 <- as.numeric(args[11])
missing_index_end1 <- as.numeric(args[12])
missing_index_start2 <- as.numeric(args[13])
missing_index_end2 <- as.numeric(args[14])

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

MCMC_interpolation_per_bipixel <- function(observed_spatial_grid, observations, k, key_location1,
                                            key_location2, cov_mod, nugget, range, smooth, nrep)
{
    id_matrix1 <- located_neighboring_pixels(observed_spatial_grid, k, key_location1)
    id_matrix2 <- located_neighboring_pixels(observed_spatial_grid, k, key_location2)
    id_matrix <- cbind(id_matrix1, id_matrix2)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    key_location <- list(s1 = c(as.numeric(key_location1$s1), as.numeric(key_location2$s1)),
                         s2 = c(as.numeric(key_location1$s2), as.numeric(key_location2$s2)))
    print(nrep)
    print(key_location)
    print(cond_coord)
    print(cond_data)
    output <- SpatialExtremes::condrmaxstab(nrep, coord = key_location,
              cond.coord = cond_coord,
              cond.data = cond_data,
              cov.mod = cov_mod, 
              nugget = nugget, 
              range = range,
              smooth = smooth)
    print(condsim)
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

alternative_MCMC_interpolation_per_bipixel_via_mask <- function(argsList)
{
    return(array(NA, dim = c(1, argsList$nrep)))
}

produce_mcmc_interpolation_per_bipixel_via_mask <- function(argsList)
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
    missing_index1 <- argsList$missing_index1
    missing_index2 <- argsList$missing_index2

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
    key_location1 <- unobserved_spatial_grid[missing_index1,]
    key_location2 <- unobserved_spatial_grid[missing_index2,]
    condsim <- MCMC_interpolation_per_bipixel(observed_spatial_grid, observations, neighbors, key_location1,
                                   key_location2, cov_mod, nugget, range, smooth, nrep)
    return(condsim)
}


produce_mcmc_interpolation_per_bipixel_via_mask_interrupted <- function(n, range, smooth, nugget, cov_mod, mask_file_name, ref_image_name,
                                                          neighbors, nrep, missing_index1, missing_index2)
{
    x <- interruptor(FUN = produce_mcmc_interpolation_per_bipixel_via_mask, args = list(n = n, range = range, smooth = smooth,
                                                                                      nugget = nugget, cov_mod = cov_mod,
                                                                                      mask_file_name = mask_file_name,
                                                                                      ref_image_name = ref_image_name,
                                                                                      neighbors = neighbors, nrep = nrep,
                                                                                      missing_index1 = missing_index1,
                                                                                      missing_index2 = missing_index2),
                                                                                      time.limit = 60, ALTFUN = alternative_MCMC_interpolation_per_bipixel_via_mask)
    return(x)
}

for(missing_index1 in missing_index_start1:missing_index_end1)
{
    for(missing_index2 in missing_index_start2:missing_index_end2)
    {
        y <- produce_mcmc_interpolation_per_bipixel_via_mask_interrupted(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                                         ref_image_name, neighbors, nrep, missing_index1,
                                                                         missing_index2)
        current_condsim_file <- paste(paste(paste(condsim_file_name, as.character(missing_index1), sep = "_"), as.character(missing_index2),
                                                    sep = "_"), "npy", sep = ".")
        np <- import("numpy")
        np$save(current_condsim_file, y)
    }
}
rm(list = ls())