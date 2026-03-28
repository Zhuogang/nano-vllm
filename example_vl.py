import os
from qwen_vl_utils import process_vision_info
from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-VL-2B-Instruct/")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # Single image + text prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Extract and preprocess images from messages
    image_inputs, _ = process_vision_info(messages)

    outputs = llm.generate(
        prompts=[messages],
        sampling_params=sampling_params,
        images=[image_inputs],
    )

    for output in outputs:
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
