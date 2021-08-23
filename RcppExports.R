cpp_contrib_RoI_Align_3D <- function(Voxel_Space, boxes, box_index) {
  .Call('_torch_cpp_contrib_RoI_Align_3D', PACKAGE = 'torchpkg', Voxel_Space, boxes, box_index)
}