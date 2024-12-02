## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(ldmppr)

## ----showdata-----------------------------------------------------------------
# Load the data
data("small_example_data")

# Display the first few rows of the dataset
nrow(small_example_data)
head(small_example_data)

## ----estimate_sc--------------------------------------------------------------
# Map the observed sizes to arrival times using the power_law_mapping function
parameter_estimation_data <- small_example_data %>%
  dplyr::mutate(time = power_law_mapping(size = size, delta = .5)) %>%
  dplyr::select(time, x, y)

# Define the grid values
x_grid <- seq(0, 25, length.out = 5)
y_grid <- seq(0, 25, length.out = 5)
t_grid <- seq(0, 1, length.out = 5)

# Define the parameter initialization values
parameter_inits <- c(1.5, 8.5, .015, 1.5, 3.2, .75, 3, .08)

# Define the upper bounds
upper_bounds <- c(1, 25, 25)

# Estimate the parameters
estimated_sc <- estimate_parameters_sc(
  data = parameter_estimation_data,
  x_grid = x_grid,
  y_grid = y_grid,
  t_grid = t_grid,
  parameter_inits = parameter_inits,
  upper_bounds = upper_bounds,
  opt_algorithm = "NLOPT_LN_SBPLX",
  nloptr_options = list(
    maxeval = 300,
    xtol_rel = 1e-3
  ),
  verbose = FALSE
)

# Obtain the optimal parameter estimates
optimal_parameters <- estimated_sc$solution
print(optimal_parameters)

## ----train_mark_model---------------------------------------------------------
# Load the example raster files
raster_paths <- list.files(system.file("extdata", package = "ldmppr"), pattern = "\\.tif$", full.names = TRUE)
rasters <- lapply(raster_paths, terra::rast)
scaled_rasters <- scale_rasters(rasters)

# Generate the time values using the power law mapping with optimal delta
model_training_data <- small_example_data %>%
  dplyr::mutate(time = power_law_mapping(size, .5))

# Train the model
example_mark_model <- train_mark_model(
  data = model_training_data,
  raster_list = scaled_rasters,
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

# Unbundle the model
example_mark_model <- bundle::unbundle(example_mark_model$bundled_model)

## ----check_model_fit, fig.width=9, fig.height=9-------------------------------
# Generate the reference pattern
reference_data <- generate_mpp(
  locations = small_example_data[, c("x", "y")],
  marks = small_example_data$size,
  xy_bounds = c(0, 25, 0, 25)
)

# Define an anchor point
M_n <- as.matrix(small_example_data[1, c("x", "y")])

# Specify the estimated parameters of the self-correcting process
estimated_parameters <- optimal_parameters

# Check the model fit
example_model_fit <- check_model_fit(
  reference_data = reference_data,
  t_min = 0,
  t_max = 1,
  sc_params = estimated_parameters,
  anchor_point = M_n,
  raster_list = scaled_rasters,
  mark_model = example_mark_model,
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

# Plot the combined global envelope test results
plot(example_model_fit$combined_env)

## ----simulate_mpp, fig.width=6.5, fig.height=6--------------------------------
# Simulate a marked point process
simulated_mpp <- simulate_mpp(
  sc_params = estimated_parameters,
  t_min = 0,
  t_max = 1,
  anchor_point = M_n,
  raster_list = scaled_rasters,
  mark_model = example_mark_model,
  xy_bounds = c(0, 25, 0, 25),
  include_comp_inds = TRUE,
  competition_radius = 10,
  correction = "none",
  thinning = TRUE
)

# Plot the simulated marked point process
plot_mpp(
  mpp_data = simulated_mpp$mpp,
  pattern_type = "simulated"
)

