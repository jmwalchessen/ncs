source("univariate_lcs_multiple_pixels.R")
source("brown_resnick_data_generation_for_R.R")
library(reticulate)

produce_mask <- function(observed_indices, n)
{
  mask <- array(0, dim = c((n**2)))
  mask[observed_indices] <- rep(1, length(observed_indices))
  return(mask)
}

produce_random_mask <- function(p, n)
{
  observed_indices <- sample(1:(n**2), floor(p*(n**2)))
  mask <- produce_mask(observed_indices, n)
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

lcs_per_pixel <- function(observed_spatial_grid, observations, k, key_location,
                          cov_mod, nugget, range, smooth, nrep)
{
  print(key_location)
  id_matrix <- located_neighboring_pixels(observed_spatial_grid, k, key_location)
  cond_data <- observations[id_matrix]
  cond_coord <- observed_spatial_grid[id_matrix,]
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

alternative_lcs_per_pixel_via_mask <- function(argsList)
{
  return(array(NA, dim = c(1, argsList$nrep)))
}

produce_lcs_per_pixel_via_mask <- function(argslist)
{
  n <- argsList$n
  range <- as.numeric(argsList$range)
  smooth <- argsList$smooth
  nugget <- argsList$nugget
  cov_mod <- argsList$cov_mod
  mask <- argsList$mask
  ref_image <- argsList$ref_image
  neighbors <- argsList$neighbors
  nrep <- argsList$nrep
  missing_index <- argsList$missing_index
  
  np <- import("numpy")
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  
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


produce_lcs_per_pixel_via_mask_interrupted <- function(n, range, smooth, nugget, cov_mod, mask, ref_image,
                                                       neighbors, nrep, missing_index)
{
  x <- interruptor(FUN = produce_lcs_per_pixel_via_mask, args = list(n = n, range = range, smooth = smooth,
                                                                     nugget = nugget, cov_mod = cov_mod,
                                                                     mask = mask,
                                                                     ref_image = ref_image,
                                                                     neighbors = neighbors, nrep = nrep,
                                                                     missing_index = missing_index),
                   time.limit = 60, ALTFUN = alternative_MCMC_interpolation_per_pixel_via_mask)
  return(x)
}

produce_empirical_pit_value_per_pixel <- function(n, range, smooth, nugget, cov_mod, mask, ref_image,
                                                  neighbors, nrep_per_pit, missing_index)
{
  mask <- flatten_matrix(mask, n)
  observed_indices <- (1:n**2)[mask == 1]
  unobserved_indices <- (1:n**2)[-observed_indices]
  observed_value <- ref_image[unobserved_indices[missing_index]]
  univariate_lcs <- produce_lcs_per_pixel_via_mask_interrupted(n, range, smooth, nugget, cov_mod, mask,
                                                               ref_image, neighbors, nrep_per_pit, missing_index)
  lcs_cdf <- ecdf(univariate_lcs)
  empirical_pit <- lcs_cdf(observed_value)
  return(empirical_pit)
}

produce_empirical_pit_values_per_pixel <- function(n, range, smooth, nugget, cov_mod, mask,
                                                   neighbors, missing_index, nrep_per_pit,
                                                   nrep, nrep_per_call, calls)
{
  ref_images <- simulate_data_across_cores(nrep_per_call, calls, nn, coord, range, smooth)
  empirical_pit_values <- c()
  for(i in 1:nrep)
  {
    empirical_pit_values[i] <- produce_empirical_pit_value_per_pixel(n, range, smooth, nugget,
                                                                     cov_mod, mask, ref_images[i,],
                                                                     neighbors, nrep_per_pit,
                                                                     missing_index)
  }
  return(empirical_pit_values)
}

produce_empirical_pit_values_multiple_pixels <- function(n, range, smooth, nugget, cov_mod, mask,
                                                         neighbors, missing_indices, nrep_per_pit,
                                                         nrep, nrep_per_call, calls)
{
  empirical_pit_values <- array(NA, dim = c(nrep))
  for(missing_index in missing_indices)
  {
    empirical_pit_values[i] <- produce_empirical_pit_values_per_pixel(n, range, smooth, nugget, cov_mod, mask,
                                                                      neighbors, missing_index, nrep_per_pit,
                                                                      nrep, nrep_per_call, calls)
  }
  return(empirical_pit_values)
}

n <- 32
range <- 1.
smooth <- 1.5
nugget <- .00001
cov_mod <- "brown"
p <- .05
mask <- produce_random_mask(p, n)
neighbors <- 7
nrep <- 1000
nrep_per_pit <- 4000
calls <- 20
nrep_per_call <- 50
np <- import("numpy")
mask <- np$load("data/model4/empirical_pit/mask1/mask.npy")
produce_empirical_pit_values_multiple_pixels(n, range, smooth, nugget, cov_mod, mask,
                                             neighbors, missing_indices, nrep_per_pit,
                                             nrep, nrep_per_call, calls)