library(reticulate)
library(MVN)


np <- import("numpy")
home_folder <- getwd()
eval_directory <- (strsplit(getwd(), "/validation")[[1]])[1]
unconditional_samples_file <- paste(eval_directory, "diffusion_generation/data/model6/ref_image2/diffusion/model6_random0_beta_min_max_01_20_1000.npy", sep = "/")
unconditional_images <- np$load(unconditional_samples_file)
unconditional_images <- unconditional_images[1:4000,,11:25,11:25]
dim(unconditional_images) <- c(4000,225)
uncond_test <- mvn(unconditional_images, subset = NULL, mvnTest = c("mardia"), covariance = TRUE,
                   tol = 1e-25, alpha = 0.5, scale = FALSE)
print(uncond_test$multivariateNormality)
  #print(uncond_test)