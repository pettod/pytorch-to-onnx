# https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

import onnx
import onnxruntime
import os
import torch
from importlib import import_module
from src.utils.utils import loadModel


MODEL_PATH = "saved_models/2024-03-12_000000"
INPUT_SHAPE = (1, 2, 256)
INPUT_NAMES = ["cnn_input"]
OUTPUT_NAMES = ["cnn_output"]


def loadModels(model_path):
    config = import_module(os.path.join(
        model_path, "codes.config").replace('/', '.')).CONFIG
    model = config.MODELS[0].to("cpu")
    loadModel(model, model_path=model_path)
    return model


def main():
    # Write ONNX file
    torch_input = torch.randn(INPUT_SHAPE)
    torch_model = loadModels(MODEL_PATH)
    onnx_filename = f"{os.path.basename(MODEL_PATH)}.onnx"
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save(onnx_filename)

    # Read ONNX file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    # Process with ONNX model
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    # Process with Pytorch model
    torch_outputs = torch_model(torch_input)
    torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)
    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    # Print results
    print("\nPyTorch and ONNX Runtime output matched!\n")
    print("ONNX output:", onnxruntime_outputs)
    print("Pytorch output:", torch_outputs)

    # This is important to write the file with this function
    torch.onnx.export(
        torch_model,
        torch_input,
        onnx_filename,    
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
    )
    print("ONNX model written to:", onnx_filename)


main()
