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
  m <- dim(observed_spatial_grid)[1]
  if(k == m)
  {
    id_matrix <- 1:m
  }
  else {
    knn <- kNN(observed_spatial_grid, k = k, query = key_location)
    id_matrix <- as.matrix(knn$id)
  }
  return(id_matrix)
}

lcs_per_pixel <- function(observed_spatial_grid, observations, k, key_location,
                          cov_mod, nugget, range, smooth, nrep)
{
    id_matrix <- located_neighboring_pixels(observed_spatial_grid, k, key_location)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
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

alternative_lcs_per_pixel_via_mask <- function(argsList)
{
    return(array(NA, dim = c(1, argsList$nrep)))
}

produce_lcs_per_pixel <- function(argsList)
{
    n <- argsList$n
    range <- as.numeric(argsList$range)
    smooth <- argsList$smooth
    nugget <- argsList$nugget
    cov_mod <- argsList$cov_mod
    mask_file_name <- argsList$mask_file_name
    ref_image_name <- argsList$ref_image_name
    neighbors <- argsList$neighbors
    nrep <- argsList$nrep
    missing_index <- argsList$missing_index
    ref_image_index <- argsList$ref_image_index

    np <- import("numpy")
    s1 <- s2 <- seq(-10, 10, length.out = n)
    s <- cbind(s1, s2)
    spatial_grid <- expand.grid(s1 = s1, 
                  s2 = s2)
    masks <- np$load(mask_file_name)
    ref_images <- exp(np$load(ref_image_name))
    mask <- masks[i,,]
    ref_image <- ref_images[i,,]
    ref_image <- flatten_matrix(ref_image, n)
    mask <- flatten_matrix(mask, n)
    observed_indices <- (1:n**2)[mask == 1]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    observations <- ref_image[observed_indices]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_observations <- ref_image[unobserved_indices]
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    key_location <- unobserved_spatial_grid[missing_index,]
    condsim <- lcs_per_pixel(observed_spatial_grid, observations, neighbors,
                             key_location, cov_mod, nugget, range, smooth, nrep)
    return(condsim)
}


produce_lcs_per_pixel_via_mask_interrupted <- function(n, range, smooth, nugget, cov_mod, mask_file_name, ref_image_name,
                                                          neighbors, nrep, missing_index, ref_image_index)
{
    arglist <- list(n = n, range = range, smooth = smooth, nugget = nugget, cov_mod = cov_mod,
                    mask_file_name = mask_file_name, ref_image_name = ref_image_name,
                    neighbors = neighbors, nrep = nrep, missing_index = missing_index,
                    ref_image_index = ref_image_index)
    x <- interruptor(FUN = produce_lcs_per_pixel_via_mask, args = arglist, time.limit = 60, ALTFUN = alternative_lcs_per_pixel_via_mask)
    return(x)
}

produce_lcs_per_image_and_mask_interrupted <- function(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                       ref_image_name, neighbors, nrep, ref_image_index)
{
  np <- import("numpy")
  masks <- np$load(mask_file_name)
  mask <- masks[ref_image_index,,]
  indices <- seq(1,n**2)
  missing_indices <- indices[mask == 0]
  condsim_matrix <- array(NA, dim = c(n**2,nrep))
  for(missing_index in missing_indices)
  {
    condsim_matrix[missing_index,] <- produce_lcs_per_pixel_via_mask_interrupted(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                           ref_image_name, neighbors, nrep, missing_index,
                                                           ref_image_index, p)
  }
  return(condsim_matrix)
}


produce_lcs <- function()
{
  n <- 32
  range <- 3.
  smooth <- 1.5
  nugget <- .00001
  cov_mod <- "brown"
  ps <- seq(.01,.05,.1,.25,.5)
  nrep <- 4000
  neighbors <- 7
  for(p in ps)
  {
    ref_folder <- paste("data/model4/random", as.character(p), sep = "")
    mask_file_name <- paste(ref_folder, "mask.npy", sep = "/")
    ref_image_name <- paste(paste(paste(ref_name, "reference_images_range_3_smooth_1.5_random", sep = "/"),
                                            as.character(p), sep = "_"), "4000.npy", sep = "_")
    condsims <- array(NA, dim = c(nrep,n**2,nrep))
    for(ref_image_index in 1:nrep)
    {

      condsims[ref_image_index,,] <- produce_lcs_per_image_and_mask_interrupted(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                                                ref_image_name, neighbors, nrep, ref_image_index)
      
    }
    condsim_file <- paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(ref_folder, "univariate_lcs_range", sep = "/"),
                                    as.character(range), sep = "_"), "smooth", sep = "_"),
                                    as.character(smooth), sep = "_"), "nugget", sep = "_"),
                                    as.character(nugget), sep = "_"), "neighbors", sep = "_"),
                                    as.character(neighbors), sep = "_"), as.character(nrep), sep = "_"), ".npy", sep = "")
    np$save(condsim_file, condsims)
  }
}

produce_lcs()




