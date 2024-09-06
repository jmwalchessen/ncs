library(reticulate)
library(MVN)



mvn(data, subset = NULL, mvnTest = c("mardia", "hz", "royston", "dh",
  "energy"), covariance = TRUE, tol = 1e-25, alpha = 0.5,
  scale = FALSE, desc = TRUE, transform = "none", R = 1000,
  univariateTest = c("SW", "CVM", "Lillie", "SF", "AD"),
  univariatePlot = "none", multivariatePlot = "none",
  multivariateOutlierMethod = "none", bc = FALSE, bcType = "rounded",
  showOutliers = FALSE, showNewData = FALSE)