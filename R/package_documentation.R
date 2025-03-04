#' SDCNN-Package: A Short Description of Your Package
#'
#' A longer description of what your package does, including key functionalities.
#'
#' @docType package
#' @name SDCNN
NULL

#' Min-max Scaling
#'
#' A detailed description of what the function does.
#'
#' @param x Description of argument `x`
#' @return A description of the return value
#' @examples
#' min_max_scale(c(1,2,3))
#' @export
min_max_scale <- function(x)
{
  low <- range(x)[1]
  high <- range(x)[2]
  out <- (x - low)/(high - low)
  return(out)
}

devtools::document()

