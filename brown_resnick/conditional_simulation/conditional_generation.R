library("dplyr")
library("parallel")
library("devtools")
library("Rlab")
library("reticulate")
working_directory <- (strsplit(getwd(), "/conditional_simulation")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)

seed_value <- 235542
set.seed(seed_value)
n <- 32
s1 <- s2 <- seq(-10, 10, length.out = n)
s <- cbind(s1, s2)
df <- (expand.grid(s1 = s1,Vs2 = s2) %>% mutate(z = c(SpatialExtremes::rmaxstab(1, coord = s, cov.mod = "brown", nugget = 0, range = 1.6, smooth = 1.6, grid = TRUE))))
ref_img <- as.array(df$z)

mask <- as.matrix(rbern(n = (n**2), p = .005))
observed_indices <- which(mask == 1)
missing_indices <- -observed_indices

k <- 100
condsim <- SpatialExtremes::condrmaxstab(k, coord = df[missing_indices, 1:2] %>% as.matrix(),
                           cond.coord = df[observed_indices,1:2] %>% as.matrix(),
                           cond.data = df[observed_indices, "z"], cov.mod = "brown",
                           nugget = 0, range = 1.6, smooth = 1.6, burnin = 1000)
condsim <- array(unlist(condsim["sim"]))
np <- import("numpy")
np$save("data/ref_img2/conditional_simulations_100.npy", condsim)
np$save("data/ref_img2/mask005.npy", mask)
np$save("data/ref_img2/ref_img.npy", ref_img)
np$save("data/ref_img2/seed_value.npy", as.array(seed_value))

