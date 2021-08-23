// cpp_contrib_torch_sort_vertices
XPtrTorchTensor cpp_contrib_torch_sort_vertices(XPtrTorchTensor vertices, XPtrTorchTensor mask, XPtrTorchTensor num_valid);
RcppExport SEXP _torch_cpp_contrib_torch_sort_vertices(SEXP verticesSEXP, SEXP maskSEXP, SEXP num_validSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter< XPtrTorchTensor >::type vertices(verticesSEXP);
  Rcpp::traits::input_parameter< XPtrTorchTensor >::type mask(maskSEXP);
  Rcpp::traits::input_parameter< XPtrTorchTensor >::type num_valid(num_validSEXP);
  rcpp_result_gen = Rcpp::wrap(cpp_contrib_torch_sort_vertices(vertices, mask, num_valid));
  return rcpp_result_gen;
  END_RCPP
}


{"_torch_cpp_contrib_torch_sort_vertices", (DL_FUNC) &_torch_cpp_contrib_torch_sort_vertices, 3},
