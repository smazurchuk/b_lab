# Binder Lab
## word2vec and other analysis

This is a repo to try and describe and document a rotation project. Central to research is **conveying** results to others in the field with nice visuals.
### Relating Sensory and Latent Vector spaces
![Image of spaces](data/vec_drawing.png)
 It is first worth looking at what if there is a linear relationship between the two spaces. This is looked at in the [notebook_plot.ipynb](notebook_plot.ipynb). In short, the assumption is that **X . M = Y** where X and Y are [535, 65] and [535, 300] respectively.

 Following a simple linear analysis, I looked into a feedforward neural network to relate the spaces.

 <img src="data/nn.png" width="300" align="middle">



I explore several models in the [relating_spaces.ipynb](relating_spaces.ipynb).

### Second Project
