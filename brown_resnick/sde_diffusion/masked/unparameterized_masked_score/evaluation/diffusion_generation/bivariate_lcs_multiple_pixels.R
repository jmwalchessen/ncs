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

order_bivariate_neighbors <- function(id_matrix)
{
  neighbors <- union(id_matrix[1,], id_matrix[2,])
  return(neighbors[1:7])
}

bivariate_located_neighboring_pixels <- function(observed_spatial_grid, k, key_location)
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
  bneighbors <- order_bivariate_neighbors(id_matrix)
  return(bneighbors)
}


bivariate_lcs_per_pixel <- function(observed_spatial_grid, observations, k, key_location1, key_location2,
                                    cov_mod, nugget, range, smooth, nrep)
{
    two_key_locations <- data.frame(s1 = c(as.numeric(key_location1$s1), as.numeric(key_location2$s1)),
                                    s2 = c(as.numeric(key_location1$s2), as.numeric(key_location2$s2)))
    id_matrix <- bivariate_located_neighboring_pixels(observed_spatial_grid, k, two_key_locations)
    print(id_matrix)
    
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    print(log(cond_data))
    output <- SpatialExtremes::condrmaxstab(nrep, coord = two_key_locations,
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

alternative_bivariate_lcs_per_pixel_via_mask <- function(argsList)
{
    return(array(NA, dim = c(1, argsList$nrep)))
}

produce_bivariate_lcs_per_pixel_via_mask <- function(argsList)
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
    unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
    key_location1 <- unobserved_spatial_grid[missing_index1,]
    key_location2 <- unobserved_spatial_grid[missing_index2,]
    condsim <- bivariate_lcs_per_pixel(observed_spatial_grid, observations, neighbors,
                                       key_location1, key_location2, cov_mod, nugget,
                                       range, smooth, nrep)
    return(condsim)
}


produce_bivariate_lcs_per_pixel_via_mask_interrupted <- function(n, range, smooth, nugget, cov_mod, mask_file_name, ref_image_name,
                                                                 neighbors, nrep, missing_index1, missing_index2)
{
    x <- interruptor(FUN = produce_bivariate_lcs_per_pixel_via_mask, args = list(n = n, range = range, smooth = smooth,
                                                                       nugget = nugget, cov_mod = cov_mod,
                                                                       mask_file_name = mask_file_name,
                                                                       ref_image_name = ref_image_name,
                                                                       neighbors = neighbors, nrep = nrep,
                                                                       missing_index1 = missing_index1,
                                                                       missing_index2 = missing_index2),
                                                                       time.limit = 60, ALTFUN = alternative_bivariate_lcs_per_pixel_via_mask)
    return(x)
}






produce_bivariate_lcs_for_multiple_pixels <- function(indices1, indices2, n, range, smooth, nugget, cov_mod,
                                                      mask_file_name, ref_image_name, neighbors, nrep,
                                                      condsim_file_name)
{
  m <- length(indices1)
  for(i in 1:m)
  {
    missing_index2 <- indices2[i]
      if((missing_index1 != missing_index2))
      {
        y <- produce_bivariate_lcs_per_pixel_via_mask_interrupted(n, range, smooth, nugget, cov_mod, mask_file_name,
                                                                ref_image_name, neighbors, nrep, missing_index1,
                                                                missing_index2)
        current_condsim_file <- paste(paste(paste(condsim_file_name, as.character(missing_index1), sep = "_"),
                                          as.character(missing_index2), sep = "_"), "npy", sep = ".")
        np <- import("numpy")
        np$save(current_condsim_file, y)
      }
  }

}

produce_bivariate_local_conditional_simulation_multiple_references <- function(indices1, indices2, n, range,
                                                                               smooth, nugget, cov_mod, neighbors,
                                                                               nrep, condsim_file_name, model_folder,
                                                                               ref_image_indices)
{
  for(i in 1:length(ref_image_indices))
  {
    ref_image_folder <- paste(paste(model_folder, "ref_image", sep = "/"), as.character(ref_image_indices[i]), sep = "")
    ref_image_name <- paste(ref_image_folder, "ref_image.npy", sep = "/")
    mask_file_name <- paste(ref_image_folder, "mask.npy", sep = "/")
    current_condsim_file_name <- paste(paste(ref_image_folder, "lcs/bivariate", sep = "/"), condsim_file_name, sep = "/")
    produce_bivariate_lcs_for_multiple_pixels(indices1, indices2, n, range, smooth, nugget, cov_mod, mask_file_name,
                                              ref_image_name, neighbors, nrep, current_condsim_file_name)
  }
}