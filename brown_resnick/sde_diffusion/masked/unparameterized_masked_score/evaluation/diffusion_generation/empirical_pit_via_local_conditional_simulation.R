library(dbscan)
library("parallel")
library("reticulate")
library(R.utils)
library(devtools)


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

produce_reference_images <- function(nsim, range, smooth, cov_mod)
{
    s1 <- seq(-10,10, length.out = n)
    s2 <- seq(-10,10, length.out = n)
    s <- cbind(s1, s2)
    working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
    spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
    devtools::install(spatialextremes_directory)
    ref_images <- SpatialExtremes::rmaxstab(nsim, coord = s, cov.mod = cov_mod, grid = TRUE, range = range, smooth = smooth)
    dim(ref_images) <- c(nsim, n, n)
    detach(SpatialExtremes, unload = TRUE)
    return(ref_images)
}

local_conditional_simulation_per_pixel <- function(observed_spatial_grid, observations, k, key_location,
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

alternative_local_conditional_simulation_per_pixel <- function(argsList)
{
    return(array(NA, dim = c(1, argsList$nrep)))
}



produce_local_conditional_simulation_per_pixel_interrupted <- function(observed_spatial_grid, observations, neighbors,
                                                                       key_location, cov_mod, nugget, range, smooth, nrep)
{
    x <- interruptor(FUN = local_conditional_simulation_per_pixel, args = list(observed_spatial_grid, observations, neighbors, key_location,
                                                                                      cov_mod, nugget, range, smooth, nrep),
                                                                                      time.limit = 60, ALTFUN = alternative_local_conditional_simulation_per_pixel)
    return(x)
}

produce_pit_value_via_local_conditional_simulation_per_fixed_location <- function(mask_file_name, missing_index, neighbors,
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
    ref_images <- produce_reference_images(nsim, range, smooth, cov_mod)
    pits <- array(dim = c(nsim))

    library(SpatialExtremes)
    for(i in 1:nsim)
    {
        ref_image <- flatten_matrix(ref_images[i,,], n)
        observations <- ref_image[observed_indices]
        unobserved_observations <- ref_image[unobserved_indices]
        unobserved_spatial_grid <- spatial_grid[unobserved_indices,]
        key_location <- unobserved_spatial_grid[missing_index,]
        conditional_simulations <- produce_local_conditional_simulation_per_pixel_interrupted(observed_spatial_grid, observations, neighbors, key_location,
                                                                                          cov_mod, nugget, range, smooth, nrep)
        empirical_cdf <- ecdf(conditional_simulations)
        pit_value <- empirical_cdf(ref_image[unobserved_index])
        pits[i] <- pit_value
    }
    detach(SpatialExtremes, unload = TRUE)
    return(pits)
}



produce_pit_values_via_local_conditional_simulation_for_multiple_pixels <- function(indices, mask_file_name, cov_mod,
                                                                                    nugget, range, smooth, nrep, nsim,
                                                                                    neighbors, pit_file_name)
{
  for(missing_index in indices)
  {
    pits <- produce_pit_value_via_local_conditional_simulation_per_fixed_location(mask_file_name, missing_index, neigbors,
                                                                                  cov_mod, nugget, range, smooth, nrep, nsim)
    print(pits[0:10])
    current_pits_file <- paste(paste(pit_file_name, as.character(missing_index), sep = "_"), "npy", sep = ".")
    np <- import("numpy")
    np$save(current_pits_file, pits)
  }
}


mask_file_name <- "data/mcmc/mask1/mask.npy"
cov_mod <- "brown"
nugget <- 0
range <- 1
smooth <- 1
nrep <- 4000
nsim <- 1000
neighbors <- 5
n <- 32
obsn <- 10
indices <- list(1,100,200,300,400,500,600,800,900,1000)
pit_file_name <- "data/mcmc/mask1/pit_values_1000_range_1_smooth_1_neighbors_5_4000.npy"
produce_random_mask(mask_file_name, obsn, n)
produce_pit_values_via_local_conditional_simulation_for_multiple_pixels(indices, mask_file_name, cov_mod,
                                                                     nugget, range, smooth, nrep, nsim,
                                                                     neighbors, pit_file_name)