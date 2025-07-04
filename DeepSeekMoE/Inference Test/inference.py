import torch
import time
from model import DeepSeekMoE, ModelArgs

def InferenceTime(model, input_ids, num_trials=100, warmup_trials=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = input_ids.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_trials):
            _ = model(input_ids)
    

    inference_times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start_time = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)

    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    std_time = (sum((t - avg_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5
    return{
        "average_ms": avg_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "std_ms": std_time
    }

def main():
    args = ModelArgs()
    model = DeepSeekMoE(args)
    batch = 1
    seq_len = 128
    input_ids = torch.randint(0, args.vocab_size, (batch, seq_len))
    
    results = InferenceTime(model, input_ids)

    print("====Inference Time Results(ms)====")
    print(f"average time: {results['average_ms']:.3f} ms")
    print(f"minimum time: {results['min_ms']:.3f} ms")
    print(f"maximum time: {results['max_ms']:.3f} ms")
    print(f"std dev: {results['std_ms']:.3f} ms")

if __name__ == "__main__":
    main()
