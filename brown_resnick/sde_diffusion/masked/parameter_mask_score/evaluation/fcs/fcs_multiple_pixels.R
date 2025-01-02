library(reticulate)
library(devtools)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

produce_mask <- function(observed_indices, n)
{
  mask <- array(0, dim = c((n**2)))
  mask[observed_indices] <- rep(1, length(observed_indices))
  return(mask)
}

produce_random_mask <- function(m, n)
{
  
  observed_indices <- sample(1:(n**2), m)
  mask <- produce_mask(observed_indices, n)
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

generate_reference_data <- function(number_of_replicates, coord, range, smooth, m, n, mask_file, ref_image_file)
{
  np <- import("numpy")
  ref_image <- brown_resnick_data_generation(number_of_replicates = 1, n = n, range = range, smooth = smooth)
  dim(ref_image) <- c(n,n)
  mask <- produce_random_mask(m,n)
  dim(mask) <- c(n,n)
  np <- import("numpy")
  np$save(mask_file, mask)
  np$save(ref_image_file, ref_image)
}


fcs <- function(mask_file_name, ref_image_name, n, nrep, range, smooth, nugget, fcs_file) 
{
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
  output <- SpatialExtremes::condrmaxstab(nrep, coord = unobserved_spatial_grid,
                                          cond.coord = observed_spatial_grid,
                                          cond.data = observations,
                                          cov.mod = "brown", 
                                          nugget = nugget, 
                                          range = range,
                                          smooth = smooth)
  condsim <- output$sim
  np$save(fcs_file, condsim)
}

m <- 7
n <- 32
ref_image_name <- "data/model4/ref_image0/ref_image.npy"
mask_file_name <- "data/model4/ref_image0/mask.npy"
nrep <- 1
range <- 1.0
smooth <- 1.5
nugget <- .00001
generate_reference_data(number_of_replicates = nrep, range = range, smooth = smooth, m = m, n = n,
                        mask_file = mask_file_name, ref_image_file = ref_image_name)
fcs_file <- "data/model4/ref_image0/fcs_range_1.0_smooth_1.5_nugget_1e5_4000.npy"
fcs(mask_file_name, ref_image_name, n, nrep, range, smooth, nugget, fcs_file)
