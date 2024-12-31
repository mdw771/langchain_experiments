import os
import io

import matplotlib.pyplot as plt


def get_api_key(name="openai"):
    with open(os.path.join(os.pardir, os.pardir, "api_keys", name + ".txt")) as f:
        key = f.readlines()[0]
    return key


def draw_graph(graph):
    img = graph.get_graph().draw_mermaid_png()
    plt.imshow(plt.imread(io.BytesIO(img), format='png'))
    plt.show()
