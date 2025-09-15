library(reticulate)


load_reference_images_masks_and_conditional_simulations <- function(p)
{

    np <- import("numpy")
    ref_folder <- paste("data/model4/random", as.character(p), sep = "")
    print(p)
    ref_images <-np$load(paste(paste(paste(ref_folder, "reference_images_range_3_smooth_1.5_random", sep = "/"),
                                as.character(p), sep = "_"), "_4000.npy", sep = ""))
    masks <- np$load(paste(ref_folder, "mask.npy", sep = "/"))
    condsims <- np$load(paste(ref_folder, "univariate_lcs_range_3.0_smooth_1.5_nugget_0.00001_neighbors_7_4000.npy", sep = "/"))
    return(list(ref_images, masks, condsims))
}

compute_empirical_pit_value <- function(p, ref_image, condsim, missing_index)
{
    ref_value <- ref_image[missing_index]
    empirical_cdf <- ecdf(condsim[missing_index,])
    empirical_pit_value <- empirical_cdf(ref_value)
    return(empirical_pit_value)
}

compute_empirical_pit_values_per_approx_percentage <- function(p)
{
   np <- import("numpy")
   ref_list <- load_reference_images_masks_and_conditional_simulations(p)
   ref_images <- ref_list[[1]] 
   masks <- ref_list[[2]]
   condsims <- ref_list[[3]]
   nrep <- 4000
   n <- 32
   ref_folder <- paste("data/model4/random", as.character(p), sep = "")
   empirical_pit_file <- paste(ref_folder, "empirical_pit_values_range_3.0_smooth_1.5_nugget_1e5_neighbors_7_4000.npy", sep = "/")
   empirical_pit_values <- array(NA, dim = c(n**2,nrep))
   for(missing_index in 1:n**2)
   {
    current_condsims <- condsims[,missing_index,]
    for(irep in 1:nrep)
    {
       ref_image <- ref_images[irep,,]
       condsim <- condsims[irep,missing_index,]
       empirical_pit_values[missing_index,irep] <- compute_empirical_pit_value(p, ref_image, condsim, missing_index) 
    }
   }
   np$save(empirical_pit_file, empirical_pit_values)  
}

compute_empirical_pit_values <- function()
{
    ps <- c(.01,.05,.1,.25,.5)
    for(p in ps)
    {
       compute_empirical_pit_values_per_approx_percentage(p) 
    }
}

compute_empirical_pit_values()