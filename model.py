import torch
from model import Transformer
from config import DEFAULT_CONFIG

def test_beam_search():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    # Load model
    model = Transformer().to(device)
    model.eval()
    
    sentence = "Ein kleiner Hund spielt im Park."
    
    print(f"\nSource (DE): {sentence}")
    
    print("Greedy translation...")
    greedy_trans = model.infer(sentence, beam_size=1)
    print(f"Greedy: {greedy_trans}")
    
    print("\nBeam Search (size=5) translation...")
    beam_trans = model.infer(sentence, beam_size=5)
    print(f"Beam 5: {beam_trans}")

if __name__ == "__main__":
    test_beam_search()
