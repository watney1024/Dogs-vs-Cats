# ONNX-MLIR Runtime API Documentation

## Introduction

The ONNX-MLIR project comes with an executable `onnx-mlir` capable of compiling ONNX models into a shared library. In this documentation, we demonstrate how to interact programmatically with the compiled shared library using ONNX-MLIR's Runtime API.

## C Runtime API

### Data Structures

- `OMTensor`: The data structure used to describe the runtime information (rank, shape, data type, etc.) associated with a tensor input or output.
  
- `OMTensorList`: The data structure used to hold a list of pointers to `OMTensor` so that they can be passed into and out of the compiled model as inputs and outputs.
  
- `OMEntryPoint`: The data structure used to return all entry point names in a model. These entry point names are the symbols of the inference functions in the model.
  
- `OMSignature`: The data structure used to return the output signature of the given entry point as a JSON string.

### Model Entry Point Signature

All compiled models will have the same exact C function signature equivalent to:

```c
OMTensorList* run_main_graph(OMTensorList*);
Intuitively, the model takes a list of tensors as input and returns a list of tensors as output.

Invoke Models Using C Runtime API
We demonstrate using the API functions to run a simple ONNX model consisting of an add operation. To create such an ONNX model, use this python script.

To compile the above model, run onnx-mlir add.onnx and a binary library add.so should appear. We can use the following C code to call into the compiled function computing the sum of two inputs:

c
复制
编辑
#include <OnnxMlirRuntime.h>
#include <stdio.h>

OMTensorList *run_main_graph(OMTensorList *);

OMTensorList *create_input_list() {
  // Shared shape & rank.
  int64_t shape[] = {3, 2};
  int64_t num_elements = shape[0] * shape[1];
  int64_t rank = 2;

  // Construct float arrays filled with 1s or 2s.
  float *x1Data = (float *)malloc(sizeof(float) * num_elements);
  for (int i = 0; i < num_elements; i++)
    x1Data[i] = 1.0;
  float *x2Data = (float *)malloc(sizeof(float) * num_elements);
  for (int i = 0; i < num_elements; i++)
    x2Data[i] = 2.0;

  // Use omTensorCreateWithOwnership "true" so float arrays are automatically
  // freed when the Tensors are destroyed.
  OMTensor *x1 = omTensorCreateWithOwnership(x1Data, shape, rank, ONNX_TYPE_FLOAT, true);
  OMTensor *x2 = omTensorCreateWithOwnership(x2Data, shape, rank, ONNX_TYPE_FLOAT, true);

  // Construct a TensorList using the Tensors
  OMTensor *list[2] = {x1, x2};
  return omTensorListCreate(list, 2);
}

int main() {
  // Generate input TensorList
  OMTensorList *input_list = create_input_list();

  // Call the compiled ONNX model function.
  OMTensorList *output_list = run_main_graph(input_list);
  if (!output_list) {
    // May inspect errno to get info about the error.
    return 1;
  }

  // Get the first tensor from the output list.
  OMTensor *y = omTensorListGetOmtByIndex(output_list, 0);
  float *outputPtr = (float *) omTensorGetDataPtr(y);

  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++)
    printf("%f ", outputPtr[i]);
  printf("\n");

  // Destroy the list and the tensors inside of it.
  // Use omTensorListDestroyShallow if only want to destroy the list themselves.
  omTensorListDestroy(input_list);
  omTensorListDestroy(output_list);
  return 0;
}
Compile with gcc main.c add.so -o add, you should see an executable add appear. Run it, and the output should be:

复制
编辑
3.000000 3.000000 3.000000 3.000000 3.000000 3.000000
Exactly as it should be.

Freeing Tensor Memory
In general, if a caller creates a tensor object (omTensorCreate), they are responsible for deallocating the data buffer separately after the tensor is destroyed. If ONNX-MLIR creates the tensor (run_main_graph), then the tensor object owns the data buffer and it is freed automatically when the tensor is destroyed.

This default behavior can be changed. When creating a tensor, a user may use omTensorCreateWithOwnership to explicitly set data buffer ownership. Additionally, after a tensor is created, omTensorSetOwning can be used to change the ownership setting.

When omTensorDestroy is called, if the ownership flag is set to "true", then the destruction of the tensor will also free any associated data buffer memory. If the ownership flag is set to "false", then the user is responsible for freeing the data buffer memory after destroying the tensor.

For tensor list objects, when omTensorListDestory is called, omTensorDestory is called on all tensors the list contained. The data buffer of each tensor is freed based on each tensor's ownership setting.

To destroy a TensorList without automatically destroying the tensors it contained, use omTensorListDestroyShallow.

Reference
For full reference to available C Runtime API, refer to include/onnx-mlir/Runtime/OMTensor.h and include/onnx-mlir/Runtime/OMTensorList.h.

pgsql
复制
编辑

This version is formatted in markdown, which is more suitable for documentation and easy reading.