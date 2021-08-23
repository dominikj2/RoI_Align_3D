// [[Rcpp::export]]
XPtrTorchTensor cpp_contrib_torch_sort_vertices (XPtrTorchTensor vertices, XPtrTorchTensor mask, XPtrTorchTensor num_valid)
{
  return XPtrTorchTensor(lantern_contrib_sort_vertices(vertices.get(), mask.get(), num_valid.get()));
}