# Question 1

# Task 1
library(readr)
data_directory <- file.path(getwd(),"IO_ProblemSets","PS_1","data","MROZ_mini.csv")
MROZ_mini <- read_csv(data_directory)

int <- matrix(rep(1, nrow(MROZ_mini)), ncol = 1)

expvar <- cbind(int, MROZ_mini$educ)

xtx <- t(expvar) %*% expvar

inv <- 1/(xtx[1,1]*xtx[2,2] - xtx[1,2]*xtx[2,1]) * matrix(c(xtx[2,2], -xtx[1,2], -xtx[2,1], xtx[1,1]), ncol = 2)

# solve(xtx)

beta_ols = inv %*% t(expvar) %*% MROZ_mini$lwage

y_ols = expvar %*% beta_ols

u_ols = MROZ_mini$lwage - y_ols

summary(lm(MROZ_mini$lwage ~ MROZ_mini$educ))


# Task 2
meanx <- 1/nrow(MROZ_mini) * sum(MROZ_mini$educ)
meany <- 1/nrow(MROZ_mini) * sum(MROZ_mini$lwage)

beta1_ols = (sum((MROZ_mini$educ-meanx)*(MROZ_mini$lwage-meany)))/(sum((MROZ_mini$educ-meanx)^2))
beta0_ols = meany - beta1_ols*meanx

# Verification
print(beta_ols)
print(beta0_ols)
print(beta1_ols)

# Task 3: Asymptotic Standard Errors

# Calculating the variance of residuals
residuals_var <- sum(u_ols^2) / (nrow(MROZ_mini) - ncol(expvar))

# Covariance matrix for the beta estimates
cov_matrix <- residuals_var * inv

# Standard errors of the coefficients
std_errors <- sqrt(diag(cov_matrix))

# Output the results
cat("Covariance matrix:\n")
print(cov_matrix)

cat("Standard errors:\n")
print(std_errors)

#Task 4
# Plotting residuals against education
plot(MROZ_mini$educ, u_ols, xlab = "Education (years)", ylab = "Residuals", main = "Residuals vs. Education")
abline(h = 0, col = "red")

#Task 5
# Test for relevance
educ_reg <- lm(educ ~ fatheduc, data = MROZ_mini)
print(summary(educ_reg))

#Task 6
# First Stage
first_stage <- lm(educ ~ fatheduc, data = MROZ_mini)
predicted_educ <- predict(first_stage, newdata = MROZ_mini)
MROZ_mini$predicted_educ <- predicted_educ

# Second Stage
second_stage <- lm(lwage ~ predicted_educ, data = MROZ_mini)
second_stage_summary <- summary(second_stage)

# Re-run OLS for comparison
ols_model <- lm(lwage ~ educ, data = MROZ_mini)
ols_summary <- summary(ols_model)

# Extracting coefficients and standard errors
results <- data.frame(
  Model = c("OLS", "IV (2SLS)"),
  Intercept = c(ols_summary$coefficients[1, 1], second_stage_summary$coefficients[1, 1]),
  Intercept_SE = c(sqrt(diag(ols_summary$cov.unscaled))[1], sqrt(diag(second_stage_summary$cov.unscaled))[1]),
  Educ_or_Pred_Educ = c(ols_summary$ccd oefficients[2, 1], second_stage_summary$coefficients[2, 1]),
  Educ_or_Pred_Educ_SE = c(sqrt(diag(ols_summary$cov.unscaled))[2], sqrt(diag(second_stage_summary$cov.unscaled))[2])
)

# show results
print(results)
