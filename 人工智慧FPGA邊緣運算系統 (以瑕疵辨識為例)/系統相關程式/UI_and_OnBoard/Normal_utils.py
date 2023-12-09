from typing import List
import cv2
import numpy as np
import vart
import time

# Image Preprocess
def preprocess_fn(image, fix_scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    image = (image-127.5)/127.5 * fix_scale
    image = image.astype(np.int8)
    return image
# Create DPU Runner
def dpu_runner(threads, subgraphs):
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    return all_dpu_runners
# Image Scaling
def input_scale(dpu):
    input_fixpos = dpu[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    return input_scale

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]
# Running Model on DPU
def runDPU(dpu,img):
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    inputData = []
    outputData = []
    inputData = np.zeros(input_ndim, dtype = np.int8, order = "C")
    inputData[0, ...] = img.reshape(input_ndim[1:])
    outputData = np.zeros(output_ndim, dtype = np.int8, order = "C")
    classes2 = ['bridge', 'normal', 'less'] # Predict Classes
    job_id = dpu.execute_async(inputData, outputData)
    out_q= np.argmax(outputData[0])
    prediction = classes2[out_q] # Prediction Result
    return  prediction

