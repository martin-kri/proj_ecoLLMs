import random
import time
import string

seed = int(time.time_ns())
rng = random.Random(seed)

def random_word(rng, length=6):
    return ''.join(rng.choice(string.ascii_lowercase) for _ in range(length))

word = random_word(rng)

print(f"Seed: {seed}")
print(f"Word: {word}")