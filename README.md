# Binder Lab
## word2vec and other analysis

This is a repo to try and describe and document a rotation project. Central to research is **conveying** results to others in the field with nice visuals.
### Relating Sensory and Latent Vector spaces
![Image of spaces](data/vec_drawing.png)
 It is first worth looking at what if there is a linear relationship between the two spaces. This is looked at in the [notebook_plot.ipynb](notebook_plot.ipynb). In short, the assumption is that **X . M = Y** where X and Y are [535, 65] and [535, 300] respectively.

 Following a simple linear analysis, I looked into a feedforward neural network to relate the spaces.
<p align="center">
 <img src="data/nn.png" width="150">
</p>


While only one hidden layer is illistrated above, I explore several models in the [relating_spaces.ipynb](relating_spaces.ipynb).

Ultimately, due to the lackluster results, I decided to take a new approach. To this end, I initially performed a t-SNE embedding, which is visualized below
<div>
    <a href="https://plot.ly/~famousshooter98/14/?share_key=BKMRi8TCCEsDlzzehgLivO" target="_blank" title="tSNE-of-Both" style="display: block; text-align: center;"><img src="https://plot.ly/~famousshooter98/14.png?share_key=BKMRi8TCCEsDlzzehgLivO" alt="tSNE-of-Both" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="famousshooter98:14" sharekey-plotly="BKMRi8TCCEsDlzzehgLivO" src="https://plot.ly/embed.js" async></script>
</div>


### Second Project
