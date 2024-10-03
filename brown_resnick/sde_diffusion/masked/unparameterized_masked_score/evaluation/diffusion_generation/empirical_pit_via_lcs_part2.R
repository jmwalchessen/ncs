library(dbscan)
library("parallel")
library("reticulate")
library(R.utils)
library(SpatialExtremes)


produce_random_mask <- function(mask_file_name, obsn, n)
{
    observed_indices <- sort(sample(1:(n**2), obsn, replace = FALSE))
    mask <- produce_mask(observed_indices, n)
    np <- import("numpy")
    np$save(mask_file_name, mask) 
}

produce_mask <- function(observed_indices, n)
{
    mask <- array(0, dim = c((n**2)))
    mask[observed_indices] <- rep(1, length(observed_indices))
    dim(mask) <- c(n,n)
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



local_conditional_simulation_per_pixel <- function(arglist)


{
    observed_spatial_grid <- arglist$observed_spatial_grid
    observations <- arglist$observations
    k <- arglist$k
    key_location <- arglist$key_location
    cov_mod <- arglist$cov_mod
    nugget <- arglist$nugget
    range <- arglist$range
    smooth <- arglist$smooth
    nrep <- arglist$nrep
    id_matrix <- located_neighboring_pixels(observed_spatial_grid, k, key_location)
    cond_data <- observations[id_matrix]
    cond_coord <- observed_spatial_grid[id_matrix,]
    output <- SpatialExtremes::condrmaxstab(nrep, coord = key_location,
              cond.coord = cond_coord,
              cond.data = cond_data,
              cov.mod = cov_mod, 
              nugget = nugget, 
              range = range,
              smooth = smooth,
              thin = 100,
              burnin = 1000)
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

alternative_local_conditional_simulation_per_pixel <- function(arglist)
{
    return(array(NA, dim = c(1, arglist$nrep)))
}



produce_local_conditional_simulation_per_pixel_interrupted <- function(observed_spatial_grid, observations, neighbors,
                                                                       key_location, cov_mod, nugget, range, smooth, nrep)
{
    x <- interruptor(FUN = local_conditional_simulation_per_pixel, args = list(observed_spatial_grid = observed_spatial_grid, observations = observations, k = neighbors,
                                                                               key_location = key_location, cov_mod = cov_mod, nugget = nugget, range = range, smooth = smooth,
                                                                               nrep = nrep),
                                                                                      time.limit = 60, ALTFUN = alternative_local_conditional_simulation_per_pixel)
    return(x)
}

produce_pit_value_via_local_conditional_simulation_per_fixed_location <- function(mask_file_name, ref_images_file_name, missing_index, neighbors,
                                                                                  cov_mod, nugget, range, smooth, nrep, nsim)
{
    n <- 32
    s1 <- s2 <- seq(-10, 10, length.out = n)
    s <- cbind(s1, s2)
    spatial_grid <- expand.grid(s1 = s1, s2 = s2)
    np <- import("numpy")
    mask <- np$load(mask_file_name)
    mask <- flatten_matrix(mask, n)
    observed_indices <- (1:n**2)[mask == 1]
    observed_spatial_grid <- spatial_grid[observed_indices,]
    unobserved_indices <- (1:n**2)[-observed_indices]
    unobserved_index <- unobserved_indices[missing_index]
    ref_images <- np$load(ref_images_file_name)
    pits <- array(dim = c(nsim))

    for(i in 1:nsim)
    {
        ref_image <- flatten_matrix(ref_images[i,,], n)
        observations <- ref_image[observed_indices]
        unobserved_observations <- ref_image[unobserved_indices]
        unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
        key_location <- unobserved_spatial_grid[missing_index,]
        conditional_simulations <- produce_local_conditional_simulation_per_pixel_interrupted(observed_spatial_grid, observations, neighbors, key_location,
                                                                                          cov_mod, nugget, range, smooth, nrep)
        if(typeof(conditional_simulations) != "character")
          {
            empirical_cdf <- ecdf(conditional_simulations)
            pit_value <- empirical_cdf(ref_image[unobserved_index])
            pits[i] <- pit_value
          }
    }
    return(pits)
}



produce_pit_values_via_local_conditional_simulation_for_multiple_pixels <- function(indices, mask_file_name, ref_images_file_name,
                                                                                    cov_mod, nugget, range, smooth, nrep, nsim,
                                                                                    neighbors, pit_file_name)
{
  for(missing_index in indices)
  {
    pits <- produce_pit_value_via_local_conditional_simulation_per_fixed_location(mask_file_name, ref_images_file_name, missing_index, neighbors,
                                                                                  cov_mod, nugget, range, smooth, nrep, nsim)
    print(pits[0:10])
    current_pits_file <- paste(paste(pit_file_name, as.character(missing_index), sep = "_"), "npy", sep = ".")
    np <- import("numpy")
    np$save(current_pits_file, pits)
  }
}


mask_file_name <- "data/mcmc/mask1/mask_range_2_smooth_1.npy"
cov_mod <- "brown"
nugget <- 0
range <- 2
smooth <- 1
nrep <- 4000
nsim <- 1000
neighbors <- 5
n <- 32
obsn <- 10
indices <- list(1,100,200,300,400,500,600,800,900,1000)
pit_file_name <- "data/mcmc/mask1/empirical_pit/pit_values_1000_range_2_smooth_1_neighbors_5_4000.npy"
ref_images_file_name <- "data/mcmc/mask1/reference_images_range_2_smooth_1.npy"
produce_pit_values_via_local_conditional_simulation_for_multiple_pixels(indices, mask_file_name, ref_images_file_name, cov_mod,
                                                                        nugget, range, smooth, nrep, nsim, neighbors, pit_file_name)