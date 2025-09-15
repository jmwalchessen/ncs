library(reticulate)
library(SpatialExtremes)


produce_mask <- function(observed_indices, n)
{
  mask <- array(0, dim = c((n**2)))
  mask[observed_indices] <- rep(1, length(observed_indices))
  return(mask)
}

produce_fixed_mask <- function(n, m)
{
  if(m == 1)
  {
  observed_indices <- c(((n**2)/2)+16)
  }
  else if(m == 2)
  {
  observed_indices <- c((((n**2)/4)+16), ((3*(n**2)/4)+16))
  }
  else
  {
    first_index <- (((n**2)/4)+16)
    second_index <- ((3*(n**2)/4)+16)
    third_index <- ((2*(n**2)/4)+16)
    fourth_index <- (((n**2)/8)+16)
    fifth_index <- ((7*(n**2)/8)+16)
    sixth_index <- ((3*(n**2)/8)+16)
    seventh_index <- ((5*(n**2)/8)+16)
    observed_indices <- c(first_index, second_index, third_index, fourth_index, fifth_index, sixth_index, seventh_index)
  }
  mask <- produce_mask(observed_indices, n)
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

brown_resnick_data_generation <- function(number_of_replicates, n, range, smooth)
{
  x <- y <- seq(-10, 10, length = n)
  coord <- expand.grid(x, y)
  y <- SpatialExtremes::rmaxstab(n = number_of_replicates, coord = coord, cov.mod = "brown",
                                 range = range, smooth = smooth)
  return(y)
}


generate_fcs <- function(n, nrep, range, smooth, nugget, m, fcs_filename) 
{
  np <- import("numpy")
  s1 <- s2 <- seq(-10, 10, length.out = n)
  s <- cbind(s1, s2)
  spatial_grid <- expand.grid(s1 = s1, 
                              s2 = s2)
  
  mask <- produce_fixed_mask(n, m)
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
                                            burnin = 1000)
  condsim <- output$sim
  np$save(fcs_filename, condsim)
}


n <- 32
nrep <- 500
range <- 3.
smooth <- 1.5
nugget <- 1e-5
m <- 7
fcs_filename <- "data/mcmc_convergence/fcs_images_burnin_1000_center_mask_obs_7_range_3.0_smooth_1.5_nugget_1e5_500_1.npy"
condsim <- generate_fcs(n, nrep, range, smooth, nugget, m, fcs_filename)