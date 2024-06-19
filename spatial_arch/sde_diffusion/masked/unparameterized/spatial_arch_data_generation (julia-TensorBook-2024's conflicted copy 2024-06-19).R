library(spdep)
library(spGARCH)
library(pracma)
library(fastmatrix)
library(reticulate)
library(parallel)

construct_norm_matrix <- function(minX, maxX, minY, maxY, n) {

    x <- seq(minX, maxX, length.out = n)
    y <- seq(minY, maxY, length.out = n)
    locations <- meshgrid(x,y)
    xlocations <- as.vector(locations$X)
    ylocations <- as.vector(locations$Y)
    X_matrix <- rep(locations$X, n**2)
    Y_matrix <- rep(locations$Y, n**2)
    dim(X_matrix) <- c(n**2,n**2)
    dim(Y_matrix) <- c(n**2,n**2)
    Xsquared = (X_matrix - t(X_matrix))*(X_matrix - t(X_matrix))
    Ysquared = (Y_matrix - t(Y_matrix))*(Y_matrix - t(Y_matrix))
    norm_matrix <- sqrt(Xsquared + Ysquared)
    return(norm_matrix)
    }

compute_bound <- function(rho, W)
{
    bound <- (matrix.norm(((rho**2)*(W %*% W)), type = "1"))**(-(1/4))
    return(bound)
}

construct_exp_kernel <- function(minX, maxX, minY, maxY, n, variance, lengthscale)
{
    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel)
}

list_d_degree_queen_neighbors <- function(labelmatrix, d, i, j)
{
    current_neighbors <- c()
    current_label <- labelmatrix[i,j]
    if (((i >= d) & (j >= d)) & (i <= (n-d)) & (j <= (n-d)))
    {
        for(k in -d:d)
        {
            for(l in -d:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i < d) & (j >= d) & (j <= (n-d)))
    {
        for(k in -i:d)
        {
            for(l in -d:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i > n - d) & (j >= d) & (j <= (n-d)))
    {
        m = n-i
        for(k in -d:m)
        {
            for(l in -d:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i >= d) & (i <= (n-d)) & (j < d))
    {
       for(k in -d:d)
        {
            for(l in -j:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        } 
    }

    else if ((i >= d) & (i <= (n-d)) & (j > (n-d)))
    {
        for(k in -d:d)
        {
            m = n-j
            for(l in -d:m)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i < d) & (j < d))
    {
        for(k in -i:d)
        {
            for(l in -j:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }

    else if ((i < d) & (j > (n-d)))
    {
        for(k in -i:d)
        {
            m = n-j
            for(l in -d:m)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }

    else if ((i > (n-d)) & (j > (n-d)))
    {
        p = n-i
        for(k in -d:p)
        {
            m = n-j
            for(l in -d:m)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i > (n-d)) & (j < d))
    {
        p = n-i
        for(k in -d:p)
        {
            for(l in -j:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    return(current_neighbors)
}

construct_d_degree_queen_contiguity_list <- function(n,d)
{
    labelmatrix <- seq(1,n**2,1)
    dim(labelmatrix) <- c(n,n)
    neighbors_list <- list()
    for(i in 1:n)
    {
        for(j in 1:n)
        { 
            current_neighbors <- list_d_degree_queen_neighbors(labelmatrix, d, i, j)
            neighbors_list[[(j-1)*n+i]] <- as.integer(current_neighbors)
        }
    }
    return(neighbors_list)
}

generate_queen_spatial_arch_process <- function(rho, alpha, n)
{
    nblist <- cell2nb(n, n, type = "queen")
    W <- nb2mat(nblist)
    y <- sim.spARCH(rho = rho, alpha = alpha, W = W)
    return(y)
}

generate_spatial_arch_processes <- function(rho, alpha, n, number_of_replicates)
{
    y_matrix <- array(0, dim = c(number_of_replicates,n**2))
    
    for(i in 1:number_of_replicates)
    {
        y_matrix[i,] <- generate_queen_spatial_arch_process(rho, alpha, n)
    }
    return(y_matrix)
}

simulate_data_per_core <- function(rho, alpha, n, number_of_replicates_per_call)
{
    y_matrix <- generate_spatial_arch_processes(rho, alpha, n,
    number_of_replicates_per_call)
}

collect_data <- function(parallel_output, nn, number_of_replicates_per_call)
{
    m <- length(parallel_output)
    y <- array(0, dim = c(number_of_replicates_per_call*(m-1), nn, nn))
    for (i in 1:(m-1))
    {
        y[((i-1)*number_of_replicates_per_call+1):(i*number_of_replicates_per_call),,] <- parallel_output[[i]]
    }
    return(y)
}

cluster_and_collect <- function(rho, alpha, n, total_number_of_replicates,
number_of_replicates_per_call)
{
    x <- y <- seq(-10, 10, length = n)
    coord <- expand.grid(x, y)
    calls <- as.integer(number_of_replicates/number_of_replicates_per_call)
    repnumberslist <- rep(number_of_replicates_per_call, calls)
    repnumberslist <- append(repnumberslist, (number_of_replicates %% number_of_replicates_per_call))
    cores <- (detectCores(logical = TRUE))
    cluster <- makeCluster(cores)
    clusterCall(cluster, function() library(spGARCH))
    clusterExport(cluster, c("n", "rho", "alpha", "simulate_data_per_core",
                         "repnumberslist", "sim.sparch", "generate_spatial_arch_processes",
                         "generate_queen_spatial_arch_process"))

    y <- parSapply(cluster, repnumberslist, function(repsnumber)
    {simulate_data_per_core(rho, alpha, n, repsnumber)})
    stopCluster(cluster)
    np <- import("numpy")
    y <- collect_data(y, nn, number_of_replicates_per_call)
    np$save("temporary_spatial_arch_samples.npy", y)
    rm(list = ls())   
}

args = commandArgs(trailingOnly=TRUE)
rho <- as.numeric(args[1])
alpha <- as.numeric(args[2])
n <- as.numeric(args[3])
number_of_replicates <- as.numeric(args[4])
seed <- as.numeric(args[5])

number_of_replicates_per_call <- 50
x <- y <- seq(-10, 10, length = n)
coord <- expand.grid(x, y)
calls <- as.integer(number_of_replicates/number_of_replicates_per_call)
repnumberslist <- rep(number_of_replicates_per_call, calls)
if ((number_of_replicates %% number_of_replicates_per_call) != 0)
{
    repnumberslist <- append(repnumberslist,
    (number_of_replicates %% number_of_replicates_per_call))
}
cores <- (detectCores(logical = TRUE))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(spdep, spGARCH))
clusterExport(cluster, c("n", "rho", "alpha", "simulate_data_per_core",
                    "repnumberslist", "generate_spatial_arch_processes",
                    "generate_queen_spatial_arch_process", "sim.spARCH"))

y <- parSapply(cluster, repnumberslist, function(repsnumber)
{simulate_data_per_core(rho, alpha, n, repsnumber)})
stopCluster(cluster)






