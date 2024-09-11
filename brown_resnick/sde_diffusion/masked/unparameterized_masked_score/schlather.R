library(SpatialExtremes)

range <- 2.2
smooth <- 1.9
nn <- 32
x <- y <- seq(-10, 10, length = nn)
s <- cbind(x, y)
coord <- as.matrix(expand.grid(x, y))
number_of_replicates <- 100
y <- rmaxstab(n = number_of_replicates, coord = s, cov.mod = "powexp",
                                   range = range, smooth = smooth, nugget = 0, grid = TRUE)

library(reticulate)
np <- import("numpy")
np$save("refimage.npy", y)