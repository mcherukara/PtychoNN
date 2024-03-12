try:
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    print('Unable to import pycuda and tensorrt. If you do not intend to use the ONNX reconstructor, ignore '
          'this message. ')
from skimage.transform import resize
import numpy as np

from ptychonn.pospred.message_logger import logger


def engine_build_from_onnx(onnx_mdl):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.max_workspace_size = 1 * (1 << 30)  # the maximum size that any layer in the network can use

    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    success = parser.parse_from_file(onnx_mdl)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        return None

    return builder.build_engine(network, config)

def mem_allocation(engine):
    """
    Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host
    inputs/outputs.
    """
    logger.debug('Expected input node shape is {}'.format(engine.get_binding_shape(0)))
    in_sz = trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size
    logger.debug('Input size: {}'.format(in_sz))
    h_input = cuda.pagelocked_empty(in_sz, dtype='float32')
    
    out_sz = trt.volume(engine.get_binding_shape(1)) * engine.max_batch_size
    h_output = cuda.pagelocked_empty(out_sz, dtype='float32')

    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    return h_input, h_output, d_input, d_output, stream

def inference(context, h_input, h_output, d_input, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Run inference.
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Synchronize the stream
    stream.synchronize()
    # Return the host
    return h_output


def transform_data_for_ptychonn(dp, target_shape, discard_len=None, overflow_correction=False):
    """
    Throw away 1/8 of the boundary region, and resize DPs to match label size.

    :param dp: np.ndarray. The data to be transformed. Can be either 3D [N, H, W] or 2D [H, W].
    :param target_shape: list[int]. The target shape.
    :param discard_len: tuple[int]. The length to discard on each side. If None, the length is default to 1/8 of the raw
                                    image size. If the numbers are negative, the images will be padded instead.
    :param overflow_correction: bool. Whether to correct overflowing pixels, whose values wrap around to the negative
                                side whn the true values surpass int16 limit.
    :return: np.ndarray.
    """
    dp = dp.astype(float)
    if overflow_correction:
        dp = correct_overflow(dp)
    if discard_len is None:
        discard_len = [dp.shape[i] // 8 for i in (-2, -1)]
    for i in (0, 1):
        if discard_len[i] > 0:
            slicer = [slice(None)] * (len(dp.shape) - 2)
            slicer_appendix = [slice(None), slice(None)]
            slicer_appendix[i] = slice(discard_len[i], -discard_len[i])
            dp = dp[tuple(slicer + slicer_appendix)]
        elif discard_len[i] < 0:
            pad_len = [(0, 0)] * (len(dp.shape) - 2)
            pad_len_appendix = [(0, 0), (0, 0)]
            pad_len_appendix[i] = (-discard_len[i], -discard_len[i])
            dp = np.pad(dp, np.array(pad_len + pad_len_appendix), mode='constant')
    target_shape = list(dp.shape[:-2]) + list(target_shape)
    if not (target_shape[-1] == dp.shape[-1] and target_shape[-2] == dp.shape[-2]):
        dp = resize(dp, target_shape, preserve_range=True, anti_aliasing=True)
    return dp


def crop_center(img, shape_to_keep=(64, 64)):
    slicer = [slice(None)] * (len(img.shape) - 2)
    for i in range(-2, 0, 1):
        st = (img.shape[i] - shape_to_keep[i]) // 2
        end = st + shape_to_keep[i]
        slicer.append(slice(st, end))
    img = img[tuple(slicer)]
    return img

def correct_overflow(arr):
    mask = arr < 0
    vals = arr[mask]
    vals = 32768 + (vals - -32768)
    arr[mask] = vals
    # logger.debug('{} overflowing values corrected.'.format(np.count_nonzero(mask)))
    return arr

