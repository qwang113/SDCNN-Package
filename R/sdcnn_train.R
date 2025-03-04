#' Train a Spatial Deep Convolutional Neural Network (SDCNN) proposed by Qi Wang, Paul A. Parker, and Robert Lund. "Spatial deep convolutional neural networks." Spatial Statistics (2025): 100883.
#'
#' This function trains an SDCNN model for spatial prediction.
#'
#' @param coords A matrix (n x 2) where the first column is longitude and the second column is latitude, and n is the number of observations.
#' @param X A matrix (n x p) of covariates, or NULL. Note that longitude and latitude has been included by default. This matrix stands for other covariates.
#' @param y A numeric vector (n x 1) of response values.
#' @param venv The name of the Conda virtual environment to use. Note only specific combination of packages is working, for the environment that the authors are using, refer to: https://github.com/qwang113/Spatial-Deep-Convolutional-Neural-Networks/blob/main/tf_gpu.yaml
#' @param basis_kernel The kernel type for basis function construction (default is "Gaussian"). Possible choices are discussed in the FRK::auto_basis() function.
#' @param pred_drop Dropout rate for prediction layers.
#' @param train_prop The proportion of data used for training (default 0.9). The rest are used to apply early stopping
#' @param model_saving_path Path for saving the model (default: current directory).
#' @param optimizer Optimization method (default: "adam"). For more options refer to keras::compile().
#' @param loss_fun Loss function for training (default: "mse"). For more options refer to keras::compile().
#' @param epoch Number of training epochs.
#' @param bat_size Batch size.
#' @return A list containing:
#' \describe{
#'   \item{model}{A trained SDCNN model (Keras model object).}
#'   \item{min_max_scale_min}{The minimum values used for min-max scaling.}
#'   \item{min_max_scale_range}{The range of values used for min-max scaling.}
#'   \item{shape_1}{A length 2 vector, represents the dimensions of the basis funcitons on the first resolution.}
#'   \item{shape_2}{A length 2 vector, represents the dimensions of the basis funcitons on the second resolution.}
#'   \item{shape_3}{A length 2 vector, represents the dimensions of the basis funcitons on the third resolution.}
#'   \item{fun_res1}{A list of the first resolution of spatial basis functions used in training.}
#'   \item{fun_res2}{A list of the second resolution of spatial basis functions used in training.}
#'   \item{fun_res3}{A list of the third resolution of spatial basis functions used in training.}
#' }
#' @examples
#' \dontrun{
#' eh_dat <- SDCNN::eh_dat
#' basis_kernel = "Gaussian"
#' venv = "tf_gpu"
#' coords <- eh_dat[,1:2]
#' y <- eh_dat[,3]
#' pred_drop <- 0.1
#' train_prop <- 0.9
#' model_saving_path <- here::here()
#' X <- NULL
#' optimizer = "adam"
#' loss_fun = "mse"
#' epoch = 1000
#' bat_size = 1000
#' model <- sdcnn_train(coords, X = NULL, y, venv, basis_kernel = "Gaussian", pred_drop = 0.1, train_prop = 0.9,model_saving_path = here::here(), optimizer = "adam", loss_fun = "mse", epoch = 10, bat_size = 1000)
#' }
#' @export
sdcnn_train <- function(coords, X = NULL, y, venv, basis_kernel = "Gaussian", pred_drop = 0.1, train_prop = 0.9,
                        model_saving_path = here::here(), optimizer = "adam", loss_fun = "mse", epoch = 10, bat_size = 1000) {
  # Specify a Conda environment
  keras::use_condaenv(venv)
  
  # Reshape the data
  dat <- data.frame(long = coords[,1], lat = coords[,2], y = y)
  sp::coordinates(dat) <- ~ long + lat
  
  # Generate basis functions
  gridbasis <- FRK::auto_basis(mainfold = FRK::plane(), data = dat, nres = 3, type = basis_kernel, regular = 1)
  res_1 <- gridbasis@df[which(gridbasis@df$res==1),]
  nbasis_1 <- nrow(res_1)
  nrow_res1 <- length(unique(res_1$loc2))
  ncol_res1 <- length(unique(res_1$loc1))
  
  res_2 <- gridbasis@df[which(gridbasis@df$res==2),]
  nbasis_2 <- nrow(res_2)
  nrow_res2 <- length(unique(res_2$loc2))
  ncol_res2 <- length(unique(res_2$loc1))
  
  res_3 <- gridbasis@df[which(gridbasis@df$res==3),]
  nbasis_3 <- nrow(res_3)
  nrow_res3 <- length(unique(res_3$loc2))
  ncol_res3 <- length(unique(res_3$loc1))
  
  print("Calculating basis functions ...")
  basis_arr_1 <- keras::array_reshape(  aperm(array(t(sapply( gridbasis@fn[1:nbasis_1], function(f) f(sp::coordinates(dat)))),
                                                    dim = c(nrow_res1, ncol_res1, nrow(dat))), c(3,2,1)),
                                        c(nrow(dat), nrow_res1, ncol_res1,1)  )
  
  
  basis_arr_2 <- keras::array_reshape(  aperm(array(t(sapply( gridbasis@fn[(nbasis_1+1):(nbasis_1+nbasis_2)], function(f) f(sp::coordinates(dat)))),
                                                    dim = c(nrow_res2, ncol_res2, nrow(dat))), c(3,2,1)),
                                        c(nrow(dat), nrow_res2, ncol_res2, 1)  )
  
  basis_arr_3 <- keras::array_reshape(  aperm(array(t(sapply( gridbasis@fn[(nbasis_1+nbasis_2+1):(nbasis_1+nbasis_2+nbasis_3)], function(f) f(sp::coordinates(dat)))),
                                                    dim = c(nrow_res3, ncol_res3, nrow(dat))), c(3,2,1)),
                                        c(nrow(dat), nrow_res3, ncol_res3, 1) )
  
  pred_drop_layer <- keras::layer_dropout(rate=pred_drop)
  
  tr_idx <- sample(1:nrow(dat), floor(train_prop*nrow(dat)))
  
  if (!is.null(X)) {
    covars_raw <- cbind(coords, X)  # Combine when X is not NULL
  } else {
    covars_raw <- coords  # Keep only coords if X is NULL
  }
  covars <- apply(covars_raw, 2, min_max_scale)
  cov_tr <- as.matrix(covars)[tr_idx,]
  cov_va <- as.matrix(covars)[-tr_idx,]
  
  # We need three convolutional input model and adding covariates.
  input_basis_1 <- keras::layer_input(shape = c(nrow_res1, ncol_res1, 1))
  input_basis_2 <- keras::layer_input(shape = c(nrow_res2, ncol_res2, 1))
  input_basis_3 <- keras::layer_input(shape = c(nrow_res3, ncol_res3, 1))
  input_cov <- keras::layer_input(shape = ncol(covars))
  
  resolution_1_conv <- input_basis_1 %>%
    keras::layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
    keras::layer_flatten() %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu')
  
  resolution_2_conv <- input_basis_2 %>%
    keras::layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_flatten() %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu')
  
  resolution_3_conv <- input_basis_3 %>%
    keras::layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_flatten() %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu')
  
  cov_model <- input_cov %>%
    keras::layer_dense(units = 100, activation = 'relu') %>% 
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu') %>%
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) %>%
    keras::layer_dense(units = 100, activation = 'relu') %>%
    keras::layer_batch_normalization() %>%
    pred_drop_layer(training = T) 
  
  
  all_model <- keras::layer_concatenate(list(resolution_1_conv, resolution_2_conv, resolution_3_conv, cov_model))
  
  output_layer <- all_model %>%  keras::layer_dense(units = 1, activation = 'linear')
  
  model_sdcnn <- keras::keras_model(inputs = list(input_basis_1, input_basis_2, input_basis_3, input_cov), outputs = output_layer)
  
  model_sdcnn <- 
    model_sdcnn %>% keras::compile(
      optimizer = optimizer,
      loss = loss_fun,
      metrics = c(loss_fun)
    )
  
  model_checkpoint <- keras::callback_model_checkpoint(
    filepath = model_saving_path,
    save_best_only = TRUE,
    monitor = "val_loss",
    mode = "min",
    verbose = 1
  )
  
  out <- model_sdcnn %>% keras::fit(
    x = list(
      keras::array_reshape(basis_arr_1[tr_idx,,,], c(length(tr_idx), nrow_res1, ncol_res1, 1)),
      keras::array_reshape(basis_arr_2[tr_idx,,,],c(length(tr_idx), nrow_res2, ncol_res2, 1)),
      keras::array_reshape(basis_arr_3[tr_idx,,,],c(length(tr_idx), nrow_res3, ncol_res3, 1)),
      cov_tr),
    y = y[tr_idx],
    epochs=epoch,
    batch_size=bat_size,
    validation_data=list(list(
      keras::array_reshape(basis_arr_1[-tr_idx,,,], c(nrow(dat) - length(tr_idx), nrow_res1, ncol_res1, 1)),
      keras::array_reshape(basis_arr_2[-tr_idx,,,], c(nrow(dat) - length(tr_idx), nrow_res2, ncol_res2, 1)),
      keras::array_reshape(basis_arr_3[-tr_idx,,,], c(nrow(dat) - length(tr_idx), nrow_res3, ncol_res3, 1)),
      cov_va), y[-tr_idx]),
    callbacks = model_checkpoint, shuffle = TRUE
  )
  return(list("model" = model_sdcnn, "min.max.scale_min" = apply(covars_raw, 2, min), "min.max.scale_range" = diff(apply(covars_raw,2, range)),
              "shape_1" = c(nrow_res1, ncol_res1), "shape_2" = c(nrow_res2, nrow_res2), "shape_3" = c(nrow_res3, nrow_res3),
              "fun_res1" = gridbasis@fn[1:nbasis_1], "fun_res2" = gridbasis@fn[(nbasis_1+1):(nbasis_1+nbasis_2)],
              "fun_res3" = gridbasis@fn[(nbasis_1+nbasis_2+1):(nbasis_1+nbasis_2+nbasis_3)]))
}
