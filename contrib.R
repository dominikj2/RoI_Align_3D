#' Contrib sort vertices
#' 
#' Based on the implementation from [roi_align_3D](https://github.com/MIC-DKFZ/medicaldetectiontoolkit/tree/master/cuda_functions/roi_align_3D)
#' 
#' @note This function does not make part of the official torch API.
#' @details All tensors should be on a CUDA device so this function can be used.
#' 
#' @param Voxel_Space A tensors.
#' @param boxes A integer tensors.
#' @param box_ind A integer tensors.

#' 
#' # crop_height, crop_width, crop_zdepth, extrapolation_value, image=Voxel_Space, boxes, box_ind ### DOM DOM DOM !!! MAY WANT TO DELETE
#' 
#' @examples
#'if (cuda_is_available()) {
#'Voxel_Space = torch_arange(1., 343)$view(c(1, 1, 7, 7,7))$'repeat'(c(2, 1, 1, 1, 1))
#'Voxel_Space[2,..] = Voxel_Space[1] +  1000

#' # for example, we have two bboxes with coords xyzxyz for xmin_ymin_zmin and ymin_ymax_zmax (first with batch_id=0, second with batch_id=1).
#' boxes = torch_tensor(c(c(2, 2, 2, 6, 6, 6),c(1, 3, 4, 4,6,7)))$view(c(2,6))
#' box_index = torch_tensor(c(1, 2), dtype=torch_int()) # index of bbox in batch
#'
#' }
#' @export
contrib_RoI_Align_3D <- function(Voxel_Space, boxes, box_ind) {
  cpp_contrib_RoI_Align_3D(Voxel_Space, boxes, box_ind) 
}

source("//home/dj806/ITCD_OS_OUTPUT/R_Code/SupComp_C_V200/ITCD_F1.R")