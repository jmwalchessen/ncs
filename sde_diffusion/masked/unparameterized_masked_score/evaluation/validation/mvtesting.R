library(reticulate)
library(MVN)


np <- import("numpy")
home_folder <- getwd()
eval_directory <- (strsplit(getwd(), "/validation")[[1]])[1]
unconditional_samples_file <- paste(eval_directory, "diffusion_generation/data/models/model6/ref_image2/model6_random0_beta_min_max_01_20_1000.npy", sep = "/")
unconditional_images <- np$load(unconditional_samples_file)

uncond_test <- mvn(data, subset = NULL, mvnTest = c("mardia", "hz", "royston", "dh",
  "energy"), covariance = TRUE, tol = 1e-25, alpha = 0.5,
  scale = FALSE, desc = TRUE, transform = "none", R = 1000,
  univariateTest = c("SW", "CVM", "Lillie", "SF", "AD"),
  univariatePlot = "none", multivariatePlot = "none",
  multivariateOutlierMethod = "none", bc = FALSE, bcType = "rounded",
  showOutliers = FALSE, showNewData = FALSE)

  print(uncond_test)