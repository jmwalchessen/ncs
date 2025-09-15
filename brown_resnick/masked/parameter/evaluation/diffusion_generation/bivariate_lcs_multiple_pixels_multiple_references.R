source("bivariate_lcs_multiple_pixels.R")


n <- 32
range_values <- seq(1.,5.,1)
smooth <- 1.5
nugget <- 1e-5
cov_mod <- "brown"
neighbors <- 7
nrep <- 4000
condsim_file_name <- "bivariate_lcs_4000_neighbors_7_nugget_1e5"
model_folder <- "data/model4"
ref_image_indices <- seq(0,4,1)
m <- 5000
indices1 <- sample(1:(n**2), m, replace = TRUE)
indices2 <- sample(1:(n**2), m, replace = TRUE)
produce_bivariate_local_conditional_simulation_multiple_references(indices1, indices2, n, range_values, smooth, nugget, cov_mod,
                                                                   neighbors, nrep, condsim_file_name, model_folder,
                                                                   ref_image_indices)