source("univariate_lcs_multiple_pixels.R")


n <- 32
indices <- seq(1,1024,1)
n <- 32
range_value <- 3.0
smooth <- 1.5
nugget <- .00001
cov_mod <- "brown"
neighbors <- 7
nrep <- 4000
condsim_file_name <- "lcs_4000_neighbors_7_nugget_1e5_missing_index"
ref_image_indices <- seq(0,4,1)
model_folder <- "data/model4"

produce_local_conditional_simulation_multiple_references(indices, n, range_values, smooth, nugget, cov_mod,
                                                         neighbors, nrep, condsim_file_name, model_folder,
                                                         ref_image_indices)