source("interpolation_functions.R")


working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
diffusion_generation_directory <- paste(working_directory, "sde_diffusion/masked/unparameterized_masked_score/evaluation/diffusion_generation", sep = "/")
np <- import("numpy")

ref_image_file <- paste(diffusion_generation_directory,
                        "data/schlather/model1/ref_image1/ref_image.npy",
                        sep = "/")
mask_file <- paste(diffusion_generation_directory, "data/schlather/model1/ref_image1/mask.npy",
                   sep = "/")
ref_image <- exp(np$load(ref_image_file))
mask <- np$load(mask_file)
n <- 32
range <- 3
smooth <- 1.6
nugget <- 0
k <- 5
nrep <- 4000
s1 <- s2 <- seq(-10, 10, length.out = n)
spatial_grid <- as.matrix(expand.grid(s1 = s1, 
                                      s2 = s2))
cov_mod <- "powexp"
flattened_mask <- mask
flattened_ref_image <- ref_image
dim(flattened_mask) <- c(n**2)
dim(flattened_ref_image) <- c(n**2)
observations <- flattened_ref_image[flattened_mask == 1]
unobserved_indices <- as.array(c(1:(n**2)))
unobserved_indices <- unobserved_indices[flattened_mask == 0]

conditional_simulations <- MCMC_interpolation(n, unobserved_indices, observations, k, cov_mod, nugget, range,
                                              smooth, nrep, spatial_grid)
mcmc_interpolation_file <- paste(diffusion_generation_directory,
"/data/schlather/model1/ref_image1/mcmc_interpolation/mcmc_interpolation_simulations_range_3_smooth_1.6_4000.npy", sep = "/")

np$save(mcmc_interpolation_file, conditional_simulations)