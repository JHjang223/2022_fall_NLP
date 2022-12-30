 #fiiting on your docker workspace
import os
import numpy as np
import tritonclient.http as httpclient

from tritonclient.utils import np_to_triton_dtype
from datasets import load_dataset

def prepare_tensor(name, input):
    tensor = httpclient.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    tensor.set_data_from_numpy(input)
    return tensor

def prepare_inputs(input0):
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
    runtime_top_k = (TOP_K * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    runtime_top_p = TOP_P * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.int32)
    is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
    beam_width = (BEAM_WIDTH * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

    inputs = [
        prepare_tensor("INPUT_0", input0_data),
        prepare_tensor("INPUT_1", output0_len),
        prepare_tensor("INPUT_2", bad_words_list),
        prepare_tensor("INPUT_3", stop_words_list),
        prepare_tensor("runtime_top_k", runtime_top_k),
        prepare_tensor("runtime_top_p", runtime_top_p),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
        prepare_tensor("temperature", temperature),
        prepare_tensor("len_penalty", len_penalty),
        prepare_tensor("repetition_penalty", repetition_penalty),
        prepare_tensor("random_seed", random_seed),
        prepare_tensor("is_return_log_probs", is_return_log_probs),
        prepare_tensor("beam_width", beam_width),
        prepare_tensor("start_id", start_ids),
        prepare_tensor("end_id", end_ids),
    ]
    return inputs

if __name__ == '__main__':
    
    URL = "localhost:8000"
    MODEl_GPT_FASTERTRANSFORMER = "ensemble" 

    OUTPUT_LEN = 128
    BATCH_SIZE = 1
    BEAM_WIDTH = 2
    TOP_K = 5
    TOP_P = 0.0

    start_id = 50256
    end_id = 50256
    
    client = httpclient.InferenceServerClient("localhost:8000",
                                           concurrency=1,
                                           verbose=False)
    
    data = load_dataset("conv_ai_2",split="train")
    per = ""
    for idx, persona in enumerate(data[0]["bot_profile"]):
        per += f"<p{idx}>" + ''.join(persona)
    
    per += "<user>"
    
    print('-----------------------------------prepared persona------------------------------ : ')
    print(per)
    print()
    print()
    print("Let's Start Conversation !! ") 
    print()
    print()
    while True:
        
        next_per = per
        input_user = input('user : ')
        if input_user == 'stop':
            break
        input_user = next_per + input_user + "<bot>"
        input0 = [[input_user],]
        inputs = prepare_inputs(input0)
        
        result = client.infer(MODEl_GPT_FASTERTRANSFORMER, inputs)

        output0 = result.as_numpy("OUTPUT_0")
        output0[0] = str(output0[0], 'utf-8').replace('<|endoftext|>', "")

        print('bot :', output0[0][len(input_user):].split('<')[0])
        next_per = input_user + output0[0][len(input_user):].split('<')[0] + "<user>"
