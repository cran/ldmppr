## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
set.seed(90210)

## ----setup--------------------------------------------------------------------
library(ldmppr)
library(dplyr)

## ----showdata-----------------------------------------------------------------
data("small_example_data")
nrow(small_example_data)
head(small_example_data)

## ----estimate_process_parameters----------------------------------------------
# Use the (x, y, size) form and let estimate_process_parameters() construct time via delta
parameter_estimation_data <- small_example_data %>%
  dplyr::select(x, y, size)

# Define the integration / grid values used by the likelihood approximation
x_grid <- seq(0, 25, length.out = 5)
y_grid <- seq(0, 25, length.out = 5)
t_grid <- seq(0, 1,  length.out = 5)

# Parameter initialization values: (alpha1, beta1, gamma1, alpha2, beta2, alpha3, beta3, gamma3)
parameter_inits <- c(1.5, 8.5, 0.015, 1.5, 3.2, 0.75, 3, 0.08)

# Upper bounds for (t, x, y)
upper_bounds <- c(1, 25, 25)


fit_sc <- estimate_process_parameters(
  data = parameter_estimation_data,
  process = "self_correcting",
  x_grid = x_grid,
  y_grid = y_grid,
  t_grid = t_grid,
  upper_bounds = upper_bounds,
  parameter_inits = parameter_inits,
  delta = 1,
  parallel = FALSE,
  strategy = c("global_local"),
  global_algorithm = "NLOPT_GN_CRS2_LM",
  local_algorithm = "NLOPT_LN_BOBYQA",
  global_options = list(maxeval = 150),
  local_options = list(maxeval = 300, xtol_rel = 1e-5, maxtime = NULL),
  verbose = FALSE
)

# Print method for ldmppr_fit objects
print(fit_sc)

# Extract parameters
estimated_parameters <- coef(fit_sc)
estimated_parameters

## ----train_mark_model---------------------------------------------------------
# Load raster covariates shipped with the package
raster_paths <- list.files(system.file("extdata", package = "ldmppr"),
  pattern = "\\.tif$", full.names = TRUE
)
raster_paths <- raster_paths[!grepl("_med\\.tif$", raster_paths)]
rasters <- lapply(raster_paths, terra::rast)

# Scale rasters once up front
scaled_rasters <- scale_rasters(rasters)

# Create training data with a time column (consistent with the delta used above)
model_training_data <- small_example_data %>%
  mutate(time = power_law_mapping(size, delta = 1))

mark_model <- train_mark_model(
  data = model_training_data,
  raster_list = scaled_rasters,
  scaled_rasters = TRUE,
  model_type = "xgboost",
  xy_bounds = c(0, 25, 0, 25),
  parallel = FALSE,
  include_comp_inds = TRUE,
  competition_radius = 10,
  correction = "none",
  selection_metric = "rmse",
  cv_folds = 5,
  tuning_grid_size = 20,
  verbose = TRUE
)

# S3 print method
mark_model

## ----save_load_mark_model, eval=FALSE-----------------------------------------
# save_path <- tempfile(fileext = ".rds")
# save_mark_model(mark_model, save_path)
# 
# mark_model2 <- load_mark_model(save_path)
# mark_model2

## ----check_model_fit, fig.width=9, fig.height=9-------------------------------
# Create a marked point pattern from the observed data
reference_mpp <- generate_mpp(
  locations = small_example_data[, c("x", "y")],
  marks = small_example_data$size,
  xy_bounds = c(0, 25, 0, 25)
)

# Anchor point (condition on an observed location)
M_n <- as.matrix(small_example_data[1, c("x", "y")])

model_check <- check_model_fit(
  reference_data = reference_mpp,
  t_min = 0,
  t_max = 1,
  sc_params = estimated_parameters,
  anchor_point = M_n,
  raster_list = scaled_rasters,
  scaled_rasters = TRUE,
  mark_model = mark_model,
  xy_bounds = c(0, 25, 0, 25),
  include_comp_inds = TRUE,
  thinning = TRUE,
  correction = "none",
  competition_radius = 10,
  n_sim = 100,
  save_sims = FALSE,
  verbose = TRUE,
  seed = 90210
)

# S3 plot method (combined global envelope test)
plot(model_check)

## ----simulate_mpp, fig.width=6.5, fig.height=6--------------------------------
simulated <- simulate_mpp(
  sc_params = estimated_parameters,
  t_min = 0,
  t_max = 1,
  anchor_point = M_n,
  raster_list = scaled_rasters,
  scaled_rasters = TRUE,
  mark_model = mark_model,
  xy_bounds = c(0, 25, 0, 25),
  include_comp_inds = TRUE,
  competition_radius = 10,
  correction = "none",
  thinning = TRUE
)

# Plot method for ldmppr_sim
plot(simulated)

# Data-frame view of the simulated realization
head(as.data.frame(simulated))

