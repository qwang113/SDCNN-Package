% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sdcnn_pred.R
\name{sdcnn_pred}
\alias{sdcnn_pred}
\title{Make Predictions Using an SDCNN Model}
\usage{
sdcnn_pred(model, coords, X = NULL, y, venv, num_pred)
}
\arguments{
\item{model}{A trained SDCNN model, must be the output from the function sdcnn_train().}

\item{coords}{A matrix (n x 2) of longitude and latitude values, where n stands for number of locations to predict.}

\item{X}{A matrix (n x p) of covariates, or NULL if not used.}

\item{y}{A numeric vector (n x 1) of response values.}

\item{venv}{The name of the Conda virtual environment to use.}

\item{num_pred}{SDCNN has uncertainty quantification. This arguments specifies the number of predictions to generate.}
}
\value{
A matrix of predictions, each column is one prediction for all locations, and each row stands for predictions for a location.
}
\description{
This function applies a trained SDCNN model to predict the response variable at unobserved locations.
}
\examples{
\dontrun{
eh_dat <- SDCNN::eh_dat
basis_kernel = "Gaussian"
venv = "tf_gpu"
coords <- eh_dat[,1:2]
y <- eh_dat[,3]
pred_drop <- 0.1
train_prop <- 0.9
model_saving_path <- here::here()
X <- NULL
optimizer = "adam"
loss_fun = "mse"
epoch = 1000
bat_size = 1000
model <- sdcnn_train(coords, X = NULL, y, venv, basis_kernel = "Gaussian",
 pred_drop = 0.1, train_prop = 0.9,model_saving_path = here::here(), optimizer = "adam", loss_fun = "mse", epoch = 10, bat_size = 1000)
preds <- sdcnn_pred(model, coords, X = NULL, y, venv, num_pred = 5)
}
}
