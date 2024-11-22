source("bivariate_lcs_multiple_pixels.R")


n <- 32
range_value <- 3.0
smooth <- 1.5
nugget <- 1e5
cov_mod <- "brown"
neighbors <- 7
nrep <- 4000
condsim_file_name <- "bivariate_lcs_4000_neighbors_7_nugget_1e5"
model_folder <- "data/model4"
ref_image_indices <- seq(0,4,1)
m <- 1000
indices1 <- sample(1:(n**2), m)
indices2 <- sample(1:(n**2), m)
produce_bivariate_local_conditional_simulation_multiple_references(indices1, indices2, n, range, smooth, nugget, cov_mod,
                                                                   neighbors, nrep, condsim_file_name, model_folder,
                                                                   ref_image_indices)