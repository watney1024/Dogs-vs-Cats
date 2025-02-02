module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "custom_rot90"} {
  func.func @main_graph(%arg0: tensor<1x3x?x?xf32> {onnx.dim_params = "2:h0,3:w0", onnx.name = "input"}) -> (tensor<?x?x?x?xf32> {onnx.dim_params = "0:Rot90AndScaleoutput_dim_0,1:Rot90AndScaleoutput_dim_1,2:w0,3:h0", onnx.name = "output"}) {
    %0 = "onnx.Custom"(%arg0) {clockwise = "yes", domain_name = "", function_name = "Rot90AndScale", k = 1 : si64, onnx_node_name = "/Rot90AndScale", scale = 1.200000e+00 : f32} : (tensor<1x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
