from pathlib import Path
import torch

from src.training.train_anc import load_checkpoint
from src.models import hard_bits_from_logits

ckpt = load_checkpoint("outputs/anc/anc_20260403T125319Z_anc_small_b32d64b0")
cfg = ckpt["config"]
alice = ckpt["models"]["alice"].eval()
bob = ckpt["models"]["bob"].eval()
eve = ckpt["models"]["eve"].eval()

plaintext = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.float32)
key = torch.tensor([[0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float32)

with torch.no_grad():
    # Alice encrypts
    ct_logits = alice(plaintext, key)
    ct_soft = torch.sigmoid(ct_logits)          
    ct_bits = hard_bits_from_logits(ct_logits)

    # Bob decrypts with the key
    bob_logits = bob(ct_soft, key)
    bob_bits = hard_bits_from_logits(bob_logits)

    # Eve tries without the key
    eve_logits = eve(ct_soft)
    eve_bits = hard_bits_from_logits(eve_logits)

print("plaintext :", plaintext.int().tolist())
print("ciphertext:", ct_bits.int().tolist())
print("bob dec   :", bob_bits.int().tolist())
print("eve guess :", eve_bits.int().tolist())