library(dbscan)
library(parallel)
library(reticulate)
library(SpatialExtremes)
library(R.utils)

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

MCMC_interpolation_per_bipixel <- function(observed_spatial_grid, observations, k, key_location1,
                                            key_location2, cov_mod, nugget, range, smooth, nrep)
{
    id_matrix1 <- located_neighboring_pixels(observed_spatial_grid, k, key_location1)
    id_matrix2 <- located_neighboring_pixels(observed_spatial_grid, k, key_location2)
    id_matrix <- cbind(id_matrix1, id_matrix2)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    key_location <- data.frame(s1 = c(as.numeric(key_location1$s1), as.numeric(key_location1$s2)),
                               s2 = c(as.numeric(key_location1$s1), as.numeric(key_location1$s2)))

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
    ref_image <- exp(np$load(ref_image_name))
    ref_image <- flatten_matrix(ref_image, n)
    mask <- flatten_matrix(mask, n)
    observed_indices <- (1:n**2)[mask == 1]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    observations <- ref_image[observed_indices]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_observations <- ref_image[unobserved_indices]
    print(log(unobserved_observations[missing_index1]))
    print(log(unobserved_observations[missing_index2]))
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    key_location1 <- unobserved_spatial_grid[missing_index1,]
    key_location2 <- unobserved_spatial_grid[missing_index2,]
    condsim <- MCMC_interpolation_per_bipixel(observed_spatial_grid, observations, neighbors, key_location1,
                                   key_location2, cov_mod, nugget, range, smooth, nrep)
    return(condsim)
}

arglist <- list(n = 32, range = 1.6, smooth = 1.6, nugget = .001, cov_mod = "brown", mask_file_name = "data/model1/ref_image1/mask.npy",
                ref_image_name = "data/model1/ref_image1/ref_image.npy", neighbors = 3, nrep = 4000, missing_index1 = 4,
                missing_index2 = 5)
condsim <- produce_mcmc_interpolation_per_bipixel_via_mask(arglist)
print(condsim[1:100,])

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