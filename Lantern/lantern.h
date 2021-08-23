LANTERN_API void* (LANTERN_PTR _lantern_contrib_RoI_Align_3D) (void* Voxel_Space, void* boxes, void* box_inde); 
HOST_API void* lantern_contrib_RoI_Align_3D(void* Voxel_Space, void* boxes, void* box_inde)
{
  void* ret = _lantern_contrib_RoI_Align_3D(Voxel_Space, boxes, box_inde);
  LANTERN_HOST_HANDLER;
  return ret;
}



LOAD_SYMBOL(_lantern_contrib_RoI_Align_3D);
