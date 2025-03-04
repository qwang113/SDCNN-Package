#' Make Predictions Using an SDCNN Model
#'
#' This function applies a trained SDCNN model to predict the response variable at unobserved locations.
#'
#' @param model A trained SDCNN model, must be the output from the function sdcnn_train().
#' @param coords A matrix (n x 2) of longitude and latitude values, where n stands for number of locations to predict.
#' @param X A matrix (n x p) of covariates, or NULL if not used.
#' @param y A numeric vector (n x 1) of response values.
#' @param venv The name of the Conda virtual environment to use.
#' @param num_pred SDCNN has uncertainty quantification. This arguments specifies the number of predictions to generate. 
#' @return A matrix of predictions, each column is one prediction, and each row stands for a location.
#' @examples
#' \dontrun{
#' load("data/eh.Rdata")
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
#' preds <- sdcnn_pred(model, coords, X = NULL, y, venv, num_pred = 5)
#' }
#' @export
sdcnn_pred <- function(model, coords, X = NULL, y, venv, num_pred) {
  predictions <- matrix(NA, nrow = nrow(coords), ncol = num_pred)
  keras::use_condaenv(venv)
  
  # Reshape the data
  dat <- data.frame(long = coords[,1], lat = coords[,2], y = y)
  sp::coordinates(dat) <- ~ long + lat
  
  # Generate basis functions
  
  nrow_res1 <- model$shape_1[1]
  ncol_res1 <- model$shape_1[2]
  nrow_res2 <- model$shape_2[1]
  ncol_res2 <- model$shape_2[2]
  nrow_res3 <- model$shape_3[1]
  ncol_res3 <- model$shape_3[2]
  
  print("Calculating basis functions ...")
  basis_arr_1 <- keras::array_reshape(  aperm(array(t(sapply( model$fun_res1, function(f) f(sp::coordinates(dat)))),
                                                    dim = c(nrow_res1, ncol_res1, nrow(dat))), c(3,2,1)),
                                        c(nrow(dat), nrow_res1, ncol_res1,1)  )
  
  basis_arr_2 <- keras::array_reshape(  aperm(array(t(sapply( model$fun_res2, function(f) f(sp::coordinates(dat)))),
                                                    dim = c(nrow_res2, ncol_res2, nrow(dat))), c(3,2,1)),
                                        c(nrow(dat), nrow_res2, ncol_res2, 1)  )
  
  basis_arr_3 <- keras::array_reshape(  aperm(array(t(sapply( model$fun_res3, function(f) f(sp::coordinates(dat)))),
                                                    dim = c(nrow_res3, ncol_res3, nrow(dat))), c(3,2,1)),
                                        c(nrow(dat), nrow_res3, ncol_res3, 1) )
  if (!is.null(X)) {
    covars <- cbind(coords, X)  # Combine when X is not NULL
  } else {
    covars <- coords  # Keep only coords if X is NULL
  }
  covars <- sweep(covars, 2, model$min.max.scale_min, FUN = "-")
  covars <- sweep(covars, 2, model$min.max.scale_range, FUN = "/")
  
  
  for (j in 1:num_pred) {
    print(j)
    predictions[,j] <- predict(model$model, list(
      basis_arr_1, basis_arr_2, basis_arr_3, as.matrix(covars)))
  }
  
  return(predictions)
}
