import numpy as np
import pandas as pd


def generate_lut(color_rgb):
    lut = np.zeros((256, 3), dtype=int)
    for i in range(256):
        lut[i] = (color_rgb * (i / 255)).astype(int)
    return lut


pink_rgb = np.array([242, 0, 255])
cyan_rgb = np.array([50, 220, 230])
white_rgb = np.array([255, 255, 255])
light_blue_rgb = np.array([0, 125, 255])
orange_rgb = np.array([255, 130, 0])

pink_lut = generate_lut(pink_rgb)
cyan_lut = generate_lut(cyan_rgb)
white_lut = generate_lut(white_rgb)
light_blue_lut = generate_lut(light_blue_rgb)
orange_lut = generate_lut(orange_rgb)

pink_lut_df = pd.DataFrame(pink_lut, columns=["Red", "Green", "Blue"])
pink_lut_df.to_csv("utils/colormaps/pink.lut", sep="\t", index_label="Index")
cyan_lut_df = pd.DataFrame(cyan_lut, columns=["Red", "Green", "Blue"])
cyan_lut_df.to_csv("utils/colormaps/cyan.lut", sep="\t", index_label="Index")
white_lut_df = pd.DataFrame(white_lut, columns=["Red", "Green", "Blue"])
white_lut_df.to_csv("utils/colormaps/white.lut", sep="\t", index_label="Index")
light_blue_lut_df = pd.DataFrame(light_blue_lut, columns=["Red", "Green", "Blue"])
light_blue_lut_df.to_csv(
    "utils/colormaps/light_blue.lut", sep="\t", index_label="Index"
)
orange_lut_df = pd.DataFrame(orange_lut, columns=["Red", "Green", "Blue"])
orange_lut_df.to_csv("utils/colormaps/orange.lut", sep="\t", index_label="Index")
