import tensorflow as tf

print("=" * 40)
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA:    {tf.test.is_built_with_cuda()}")
print("=" * 40)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"\nGPU detectada! ({len(gpus)} dispositivo(s))\n")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
else:
    print("\nNenhuma GPU detectada. Rodando apenas na CPU.")

print("=" * 40)
