from tqdm import tqdm, trange
from time import sleep
from random import randrange, uniform

pbar = tqdm(total=100, bar_format="{l_bar}{bar:20}{r_bar}"
            )
for i in range(10):
    postfix_dict = {
        'Loss': f"{uniform(0, 3):.5f}",
        'Accuracy': f"{uniform(0, 100):.2f}%"
    }
    sleep(0.3)
    pbar.set_postfix(postfix_dict)
    pbar.set_description_str(f'Training Epoch {i} | Loss = {uniform(0, 3):.5f}, Accuracy = {uniform(0, 100):.2f}%')
    pbar.update(10)
pbar.close()