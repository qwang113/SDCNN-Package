#' Min-Max Scaling
#'
#' This function normalizes a numeric vector to the range [0,1].
#'
#' @param x A numeric vector.
#' @return A normalized numeric vector.
#' @examples
#' min_max_scale(c(1,2,3))
#' @export
min_max_scale <- function(x) {
  low <- min(x)
  high <- max(x)
  out <- (x - low) / (high - low)
  return(out)
}