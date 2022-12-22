# Data Organization
shrooms <- read_csv("data/mushrooms.csv")

shrooms <-
  shrooms %>%
  janitor::clean_names()

## class
shrooms$class[shrooms$class == "p"] <- "poisonous"
shrooms$class[shrooms$class == "e"] <- "edible"

## cap_shape
shrooms$cap_shape[shrooms$cap_shape == "b"] <- "bell"
shrooms$cap_shape[shrooms$cap_shape == "c"] <- "conical"
shrooms$cap_shape[shrooms$cap_shape == "x"] <- "convex"
shrooms$cap_shape[shrooms$cap_shape == "f"] <- "flat"
shrooms$cap_shape[shrooms$cap_shape == "k"] <- "knobbed"
shrooms$cap_shape[shrooms$cap_shape == "s"] <- "sunken"

## cap_surface
shrooms$cap_surface[shrooms$cap_surface == "f"] <- "fibrous"
shrooms$cap_surface[shrooms$cap_surface == "g"] <- "grooves"
shrooms$cap_surface[shrooms$cap_surface == "y"] <- "scaly"
shrooms$cap_surface[shrooms$cap_surface == "s"] <- "smooth"

## cap_color
shrooms$cap_color[shrooms$cap_color == "n"] <- "brown"
shrooms$cap_color[shrooms$cap_color == "b"] <- "buff"
shrooms$cap_color[shrooms$cap_color == "c"] <- "cinnamon"
shrooms$cap_color[shrooms$cap_color == "g"] <- "gray"
shrooms$cap_color[shrooms$cap_color == "r"] <- "green"
shrooms$cap_color[shrooms$cap_color == "p"] <- "pink"
shrooms$cap_color[shrooms$cap_color == "u"] <- "purple"
shrooms$cap_color[shrooms$cap_color == "e"] <- "red"
shrooms$cap_color[shrooms$cap_color == "w"] <- "white"
shrooms$cap_color[shrooms$cap_color == "y"] <- "yellow"

## bruises
shrooms$bruises[shrooms$bruises == "TRUE"] <- "bruises"
shrooms$bruises[shrooms$bruises == "FALSE"] <- "no bruises"

## odor
shrooms$odor[shrooms$odor == "a"] <- "almond"
shrooms$odor[shrooms$odor == "l"] <- "anise"
shrooms$odor[shrooms$odor == "c"] <- "creosote"
shrooms$odor[shrooms$odor == "y"] <- "fishy"
shrooms$odor[shrooms$odor == "f"] <- "foul"
shrooms$odor[shrooms$odor == "m"] <- "musty"
shrooms$odor[shrooms$odor == "n"] <- "none"
shrooms$odor[shrooms$odor == "p"] <- "pungent"
shrooms$odor[shrooms$odor == "s"] <- "spicy"

## gill_attachment
shrooms$gill_attachment[shrooms$gill_attachment == "a"] <- "attached"
shrooms$gill_attachment[shrooms$gill_attachment == "d"] <- "descending"
shrooms$gill_attachment[shrooms$gill_attachment == "f"] <- "free"
shrooms$gill_attachment[shrooms$gill_attachment == "n"] <- "notched"

## gill_spacing
shrooms$gill_spacing[shrooms$gill_spacing == "c"] <- "close"
shrooms$gill_spacing[shrooms$gill_spacing == "w"] <- "crowded"
shrooms$gill_spacing[shrooms$gill_spacing == "d"] <- "distant"

## gill_size
shrooms$gill_size[shrooms$gill_size == "b"] <- "broad"
shrooms$gill_size[shrooms$gill_size == "n"] <- "narrow"

## gill_color
shrooms$gill_color[shrooms$gill_color == "k"] <- "black"
shrooms$gill_color[shrooms$gill_color == "n"] <- "brown"
shrooms$gill_color[shrooms$gill_color == "b"] <- "buff"
shrooms$gill_color[shrooms$gill_color == "h"] <- "chocolate"
shrooms$gill_color[shrooms$gill_color == "g"] <- "gray"
shrooms$gill_color[shrooms$gill_color == "r"] <- "green"
shrooms$gill_color[shrooms$gill_color == "o"] <- "orange"
shrooms$gill_color[shrooms$gill_color == "p"] <- "pink"
shrooms$gill_color[shrooms$gill_color == "u"] <- "purple"
shrooms$gill_color[shrooms$gill_color == "e"] <- "red"
shrooms$gill_color[shrooms$gill_color == "w"] <- "white"
shrooms$gill_color[shrooms$gill_color == "y"] <- "yellow"

## stalk_shape
shrooms$stalk_shape[shrooms$stalk_shape == "e"] <- "enlarging"
shrooms$stalk_shape[shrooms$stalk_shape == "t"] <- "tapering"

## stalk_root
shrooms$stalk_root[shrooms$stalk_root == "b"] <- "bulbous"
shrooms$stalk_root[shrooms$stalk_root == "c"] <- "club"
shrooms$stalk_root[shrooms$stalk_root == "u"] <- "cup"
shrooms$stalk_root[shrooms$stalk_root == "e"] <- "equal"
shrooms$stalk_root[shrooms$stalk_root == "z"] <- "rhizomorphs"
shrooms$stalk_root[shrooms$stalk_root == "r"] <- "rooted"
shrooms$stalk_root[shrooms$stalk_root == "?"] <- NA

## stalk_surface_above_ring
shrooms$stalk_surface_above_ring[shrooms$stalk_surface_above_ring == "f"] <- "fibrous"
shrooms$stalk_surface_above_ring[shrooms$stalk_surface_above_ring == "y"] <- "scaly"
shrooms$stalk_surface_above_ring[shrooms$stalk_surface_above_ring == "k"] <- "silky"
shrooms$stalk_surface_above_ring[shrooms$stalk_surface_above_ring == "s"] <- "smooth"

## stalk_surface_below_ring
shrooms$stalk_surface_below_ring[shrooms$stalk_surface_below_ring == "f"] <- "fibrous"
shrooms$stalk_surface_below_ring[shrooms$stalk_surface_below_ring == "y"] <- "scaly"
shrooms$stalk_surface_below_ring[shrooms$stalk_surface_below_ring == "k"] <- "silky"
shrooms$stalk_surface_below_ring[shrooms$stalk_surface_below_ring == "s"] <- "smooth"

## stalk_color_above_ring
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "n"] <- "brown"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "b"] <- "buff"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "c"] <- "cinnamon"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "g"] <- "gray"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "o"] <- "orange"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "p"] <- "pink"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "e"] <- "red"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "w"] <- "white"
shrooms$stalk_color_above_ring[shrooms$stalk_color_above_ring == "y"] <- "yellow"

## stalk_color_below_ring
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "n"] <- "brown"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "b"] <- "buff"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "c"] <- "cinnamon"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "g"] <- "gray"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "o"] <- "orange"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "p"] <- "pink"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "e"] <- "red"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "w"] <- "white"
shrooms$stalk_color_below_ring[shrooms$stalk_color_below_ring == "y"] <- "yellow"

## veil_type
shrooms$veil_type[shrooms$veil_type == "p"] <- "partial"
shrooms$veil_type[shrooms$veil_type == "u"] <- "universal"

## veil_color
shrooms$veil_color[shrooms$veil_color == "n"] <- "brown"
shrooms$veil_color[shrooms$veil_color == "o"] <- "orange"
shrooms$veil_color[shrooms$veil_color == "w"] <- "white"
shrooms$veil_color[shrooms$veil_color == "y"] <- "yellow"

## ring_number
shrooms$ring_number[shrooms$ring_number == "n"] <- "none"
shrooms$ring_number[shrooms$ring_number == "o"] <- "one"
shrooms$ring_number[shrooms$ring_number == "t"] <- "two"

## ring_type
shrooms$ring_type[shrooms$ring_type == "c"] <- "cobwebby"
shrooms$ring_type[shrooms$ring_type == "e"] <- "evanescent"
shrooms$ring_type[shrooms$ring_type == "f"] <- "flaring"
shrooms$ring_type[shrooms$ring_type == "l"] <- "large"
shrooms$ring_type[shrooms$ring_type == "n"] <- "none"
shrooms$ring_type[shrooms$ring_type == "p"] <- "pendant"
shrooms$ring_type[shrooms$ring_type == "s"] <- "sheathing"
shrooms$ring_type[shrooms$ring_type == "z"] <- "zone"

## spore_print_color
shrooms$spore_print_color[shrooms$spore_print_color == "k"] <- "black"
shrooms$spore_print_color[shrooms$spore_print_color == "n"] <- "brown"
shrooms$spore_print_color[shrooms$spore_print_color == "b"] <- "buff"
shrooms$spore_print_color[shrooms$spore_print_color == "h"] <- "chocolate"
shrooms$spore_print_color[shrooms$spore_print_color == "r"] <- "green"
shrooms$spore_print_color[shrooms$spore_print_color == "o"] <- "orange"
shrooms$spore_print_color[shrooms$spore_print_color == "u"] <- "purple"
shrooms$spore_print_color[shrooms$spore_print_color == "w"] <- "white"
shrooms$spore_print_color[shrooms$spore_print_color == "y"] <- "yellow"

## population
shrooms$population[shrooms$population == "a"] <- "abundant"
shrooms$population[shrooms$population == "c"] <- "clustered"
shrooms$population[shrooms$population == "n"] <- "numerous"
shrooms$population[shrooms$population == "s"] <- "scattered"
shrooms$population[shrooms$population == "v"] <- "several"
shrooms$population[shrooms$population == "y"] <- "solitary"

## habitat
shrooms$habitat[shrooms$habitat == "g"] <- "grasses"
shrooms$habitat[shrooms$habitat == "l"] <- "leaves"
shrooms$habitat[shrooms$habitat == "m"] <- "meadows"
shrooms$habitat[shrooms$habitat == "p"] <- "paths"
shrooms$habitat[shrooms$habitat == "u"] <- "urba"
shrooms$habitat[shrooms$habitat == "w"] <- "waste"
shrooms$habitat[shrooms$habitat == "d"] <- "woods"

## checking for missing variables
shrooms %>%
  skimr::skim_without_charts()

## checking distribution of class
ggplot(shrooms, aes(class)) +
  geom_bar()





# Functions Developed for Graphing Data

## Category Graph
category_graph <- function(column) {
  arg <- match.call()
  pars <- as.list(match.call()[-1])
  column_name <- as.character(pars$column)
  data <- shrooms %>% select(class, column_name)
  data <- data[complete.cases(data),]
  count <- count(data, data[,1:2]) %>% pivot_wider(names_from = class, values_from = n)
  count[is.na(count)] <- 0
  calculations <- mutate(count, total = edible + poisonous) %>%
    mutate(edible = edible / total) %>%
    mutate(poisonous = poisonous / total)
  graph <- calculations[,1:3] %>%
    pivot_longer(c(edible, poisonous), names_to = "class", values_to = "proportion") %>%
    ggplot(aes(eval(arg$column), proportion, fill = class)) +
    geom_col() +
    xlab(column_name)
  return(graph)
}

## Class Graph
class_graph <- function(column) {
  arg <- match.call()
  pars <- as.list(match.call()[-1])
  column_name <- as.character(pars$column)
  data <- shrooms %>% select(class, column_name)
  data <- data[complete.cases(data),]
  count <- count(data, data[,1:2]) %>% pivot_wider(names_from = class, values_from = n)
  count[is.na(count)] <- 0
  calculations <- mutate(count, edible_count = sum(edible)) %>%
    mutate(poisonous_count = sum(poisonous)) %>%
    mutate(edible = edible / edible_count) %>%
    mutate(poisonous = poisonous / poisonous_count)
  graph <- calculations[,1:3] %>%
    pivot_longer(c(edible, poisonous), names_to = "class", values_to = "proportion") %>%
    ggplot(aes(class, proportion, fill = eval(arg$column))) +
    geom_col() + 
    scale_fill_discrete(column_name)
  return(graph)
}





# Splitting 
shrooms <- mutate(shrooms, class = factor(class))

shrooms_split <- initial_split(shrooms, prop = 0.75, strata = class)           # should divide into 6093 x 23 and 2031 x 23 tibbles
shrooms_train <- training(shrooms_split)
shrooms_test <- testing(shrooms_split)

## has correct proportion of rows from initial data (6093 and 2031)
nrow(shrooms_train)
nrow(shrooms_test)

## has correct number of columns (same as initial data = 23)
ncol(shrooms_train)
ncol(shrooms_test)

## folds and keep predictions
shrooms_folds <- vfold_cv(shrooms_train, v = 10, repeats = 5)
keep_pred_resamples <- control_resamples(save_pred = TRUE, save_workflow = TRUE)
keep_pred_grid <- control_grid(save_pred = TRUE, save_workflow = TRUE)






# Recipes
simple_rec <- 
  recipe(class ~ ., data = shrooms_train) %>%                   # utilizes all variables to predict class
  step_string2factor(all_nominal_predictors()) %>%              # changes character variables to factors
  step_impute_mode(all_nominal_predictors()) %>%                # predicts/accounts for missing stalk_root values
  step_novel(all_nominal_predictors()) %>%                      # handles factors with folding
  step_dummy(all_nominal_predictors()) %>%                      # dummy encodes categorical variables
  step_nzv(all_predictors(), freq_cut = 90/10)                  # removes anything with near-zero variance, specified
simple_rec %>%                                                  # checking recipe, now has 40 columns
  prep() %>%
  bake(new_data = shrooms_train) %>%
  view()

tuning_rec <- 
  recipe(class ~ ., data = shrooms_train) %>%                   # utilizes all variables to predict class
  step_string2factor(all_nominal_predictors()) %>%              # changes character variables to factors
  step_impute_mode(all_nominal_predictors()) %>%                # predicts/accounts for missing stalk_root values
  step_novel(all_nominal_predictors()) %>%                      # handles factors with folding
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%      # one-hot encodes categorical variables
  step_nzv(all_predictors(), freq_cut = 90/10)                  # removes anything with near-zero variance, specified
tuning_rec %>%                                                  # checking recipe, now has 50 columns
  prep() %>%
  bake(new_data = shrooms_train) %>%
  view()




# Null Model 

## Model Specification
null_spec <-
  null_model() %>% 
  set_engine("parsnip") %>% 
  set_mode("classification")

## Workflow
null_wflow <- 
  workflow() %>% 
  add_model(null_spec) %>% 
  add_recipe(simple_rec)

## Fitting to Resamples
null_fit <- null_wflow %>% 
  fit_resamples(
    resamples = shrooms_folds,
    control = keep_pred_resamples
  )
save(null_fit, file = "results/null_fit.rda")





# Logistic Regression Model

## Model Specification
log_spec <-
  logistic_reg() %>%
  set_engine('glm')

## Workflow
log_wflow <- 
  workflow() %>% 
  add_model(log_spec) %>% 
  add_recipe(simple_rec)

## Fitting to Resamples
log_fit <- log_wflow %>%
  fit_resamples(
    resamples = shrooms_folds,
    control = keep_pred_resamples
  )
save(log_fit, file = "results/log_fit.rda")





# Random Forest Model

## Model Specification
rf_spec <-
  rand_forest(mtry = tune(), min_n = tune(), trees = 1500) %>%
  set_engine('ranger') %>%
  set_mode('classification')

## Workflow
rf_wflow <- 
  workflow() %>% 
  add_model(rf_spec) %>% 
  add_recipe(tuning_rec)

## Tuning Parameter Grid
rf_params <- parameters(rf_spec) %>% 
  update(mtry = mtry(range = c(2, 25)))                             # uses 2-25 (around 1/2) of the predictor variables
rf_grid <- grid_regular(rf_params, levels = 5)

## Fitting to Resamples
rf_fit <-
  rf_wflow %>%
  tune_grid(
    resamples = shrooms_folds,
    control = keep_pred_grid,
    grid = rf_grid
  )
save(rf_fit, file = "results/rf_fit.rda")






# Boosted Tree Model

## Model Specification
bt_spec <-
  boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

## Workflow
bt_wflow <- 
  workflow() %>% 
  add_model(bt_spec) %>%
  add_recipe(tuning_rec)

## Tuning Parameter Grid
bt_params <- parameters(bt_spec) %>% 
  update(mtry = mtry(range = c(2, 25))) %>%                   # uses 2-25 (around 1/2) of the predictor variables
  update(learn_rate = learn_rate(range = c(-5, -0.2)))
bt_grid <- grid_regular(bt_params, levels = 5)

## Fitting to Resamples
bt_fit <-
  bt_wflow %>%
  tune_grid(
    resamples = shrooms_folds,
    control = keep_pred_grid,
    grid = bt_grid
  )
save(bt_fit, file = "results/bt_fit.rda")






# k-Nearest Neighbors Model

## Model Specification
k_spec <-
  nearest_neighbor(neighbors = tune()) %>%
  set_engine('kknn') %>%
  set_mode('classification')

## Workflow
k_wflow <-
  workflow() %>% 
  add_model(k_spec) %>%
  add_recipe(tuning_rec)

## Tuning Parameter Grid
k_params <- parameters(k_spec) %>%
  update(neighbors = neighbors(range = c(5, 37)))
k_grid <- grid_regular(k_params, levels = 5)

## Fitting to Resamples
k_fit <-
  k_wflow %>%
  tune_grid(
    resamples = shrooms_folds, 
    control = keep_pred_grid,
    grid = k_grid
  )
save(k_fit, file = "results/k_fit.rda")






# Selecting the Best Model

## load files
load(file = "results/null_fit.rda")
load(file = "results/log_fit.rda")
load(file = "results/rf_fit.rda")
load(file = "results/bt_fit.rda")
load(file = "results/k_fit.rda")

## Model Set
model_set <-
  as_workflow_set(
    "null" = null_fit,
    "log" = log_fit,
    "rf" = rf_fit,
    "bt" = bt_fit,
    "k" = k_fit
  )

model_set %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean)) %>%
  select(wflow_id, mean, std_err, n)

## visual of results
model_set %>%
  autoplot(metric = "accuracy")

## select best
show_best(log_fit, metric = "accuracy")
show_best(rf_fit, metric = "accuracy")
show_best(bt_fit, metric = "accuracy")
show_best(k_fit, metric = "accuracy")




# Fitting Model to the Testing Set

## Fitting the Best Model to the Training Set
log_spec <-
  logistic_reg() %>%
  set_engine('glm')
log_wflow <- 
  workflow() %>% 
  add_model(log_spec) %>% 
  add_recipe(simple_rec)

log_fit_final <- fit(log_wflow, shrooms_train)

## Looking at Accuracy
log_pred <- predict(log_fit_final, new_data = shrooms_test)

predicted_values <-
  shrooms_test %>%
  select(class) %>%
  bind_cols(log_pred)
predicted_values

log_acc <- accuracy(predicted_values, class, .pred_class)
log_acc

## Confusion Matrix Confirming Accuracy
conf_mat(predicted_values, class, .pred_class)
