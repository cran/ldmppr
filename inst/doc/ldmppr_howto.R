## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
set.seed(90210)
library(ldmppr)
library(dplyr)

## ----showdata-----------------------------------------------------------------
data("small_example_data")
nrow(small_example_data)
head(small_example_data)

## ----estimate_process_parameters----------------------------------------------
# Use the (x, y, size) form and let estimate_process_parameters() construct time via delta
parameter_estimation_data <- small_example_data

# Upper bounds for (t, x, y)
ub <- c(1, 25, 25)

# Define the integration / grid schedule used by the likelihood approximation
grids <- ldmppr_grids(
  upper_bounds = ub,
  levels = list(
    c(20, 20, 20)
    )
)

# Define optimizer budgets/options for the global + local stages
budgets <- ldmppr_budgets(
  global_options = list(
    maxeval = 150
    ),
  local_budget_first_level = list(
    maxeval = 300,
    xtol_rel = 1e-5
    ),
  local_budget_refinement_levels = list(
    maxeval = 300,
    xtol_rel = 1e-5
    )
)

# Parameter initialization values: (alpha1, beta1, gamma1, alpha2, beta2, alpha3, beta3, gamma3)
parameter_inits <- c(1.5, 8.5, 0.015, 1.5, 3.2, 0.75, 3, 0.08)

fit_sc <- estimate_process_parameters(
  data = parameter_estimation_data,
  grids = grids,
  budgets = budgets,
  parameter_inits = parameter_inits,
  delta = 1,
  rescore_control = list(
    enabled = TRUE,
    top = 5L,
    objective_tol = 1e-6,
    param_tol = 0.10
  ),
  parallel = FALSE,
  strategy = "global_local",
  global_algorithm = "NLOPT_GN_CRS2_LM",
  local_algorithm = "NLOPT_LN_BOBYQA",
  verbose = TRUE
)
# Print method for ldmppr_fit objects
fit_sc

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

# Train the mark model, passing the estimated self-correcting model fit from Step 1
mark_model <- train_mark_model(
  data = fit_sc,
  raster_list = scaled_rasters,
  scaled_rasters = TRUE,
  model_type = "xgboost",
  parallel = FALSE,
  include_comp_inds = FALSE,
  competition_radius = 10,
  edge_correction = "none",
  selection_metric = "rmse",
  cv_folds = 5,
  tuning_grid_size = 20,
  verbose = TRUE
)

# Print method for ldmppr_mark_model objects
print(mark_model)

# Summary method for ldmppr_mark_model objects
summary(mark_model)

## ----save_load_mark_model, eval=FALSE-----------------------------------------
# save_path <- tempfile(fileext = ".rds")
# save_mark_model(mark_model, save_path)
# 
# mark_model2 <- load_mark_model(save_path)
# mark_model2

## ----check_model_fit, fig.width=9, fig.height=9, warning=FALSE----------------
# Check the model fit by passing the estimated process and trained mark models from Steps 1 and 2
model_check <- check_model_fit(
  t_min = 0,
  t_max = 1,
  process = "self_correcting",
  process_fit = fit_sc,
  mark_model = mark_model,
  include_comp_inds = FALSE,
  thinning = TRUE,
  edge_correction = "none",
  competition_radius = 10,
  n_sim = 199,
  mark_mode = "mark_model",
  fg_correction = "km",
  save_sims = FALSE,
  verbose = TRUE,
  seed = 90210
)

# Plot method for ldmppr_model_check objects (defaults to combined global envelope test)
plot(model_check)

## ----simulate_mpp, fig.width=6.5, fig.height=6--------------------------------
# Simulate a new marked point pattern by passing the estimated process and trained mark models from Steps 1 and 2
simulated <- simulate_mpp(
  process = "self_correcting",
  process_fit = fit_sc,
  t_min = 0,
  t_max = 1,
  mark_model = mark_model,
  mark_mode = "mark_model",
  include_comp_inds = TRUE,
  competition_radius = 10,
  edge_correction = "none",
  thinning = TRUE,
  seed = 90210
)

# Plot method for ldmppr_sim
plot(simulated)

# Data-frame view of the simulated realization
head(as.data.frame(simulated))

