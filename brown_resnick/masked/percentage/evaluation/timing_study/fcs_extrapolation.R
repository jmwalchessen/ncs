library(reticulate)


np <- import("numpy")
range_values <- c(1,2,3,4,5)

regression_per_range <- function(range_value, fcs_type)
{
    fcs_file <- paste(paste(paste(paste("data/range", as.character(range_value), sep = "_"),
                "fcs", sep = "_"), fcs_type, sep = "_"), "timing_azure_gpu_1_7_tnrep_50.npy",
                sep = "_")
    fcs_times <- np$load(fcs_file)
    avg_fcs_times <- array(NA, dim = c(7))
    avg_div_times <- array(NA, dim = c(7))
    for(i in 1:7)
    {
        avg_fcs_times[i] <- mean(fcs_times[i,])
        avg_div_times[i] <- avg_fcs_times[i]/avg_fcs_times[1]
    }
    df <- data.frame(locations = seq(1,7), div = log(avg_div_times))
    linear_model <- lm(div~locations, df)
    return(linear_model)

}


exp_regression_per_range <- function(range_value, fcs_type)
{
    fcs_file <- paste(paste(paste(paste("data/range", as.character(range_value), sep = "_"),
                "fcs", sep = "_"), fcs_type, sep = "_"), "timing_azure_gpu_1_7_tnrep_50.npy",
                sep = "_")
    fcs_times <- np$load(fcs_file)
    avg_fcs_times <- array(NA, dim = c(7))
    avg_div_times <- array(NA, dim = c(7))
    for(i in 1:7)
    {
        avg_fcs_times[i] <- mean(fcs_times[i,])
        avg_div_times[i] <- avg_fcs_times[i]/avg_fcs_times[1]
    }
    df <- data.frame(locations = seq(1,7), div = avg_div_times)
    linear_model <- lm(log(div)~locations, df)
    return(linear_model)
}

range_value <- 2.
fcs_type <- "user"
lin <- exp_regression_per_range(range_value, fcs_type)