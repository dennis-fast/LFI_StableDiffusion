
How good is textual inversion trained Stable Diffusion 2 at generating new realistic Cézanne landscape paintings? Does a fine-tuned Convolutional Neural Network classify their style as Original, Replica, Stable Diffusion image w/o textual inversion, or General Impressionist Landscape Painting? \
\
The dataset to fine-tune the CNN consists of a range of landscape paintings similar in style to that of Paul Cézanne. The photographs of the images are unfortunately not consistent in quality and were taken from a range of different sources. The images were curated and divided into 5 categories from most (0) to least similar (4) in style.\
\
0 - authentic Cézanne landscape paintings - 89 images\
1 - hand-painted replicas and forgeries of authentic Cézanne landscape paintings - 68 images\
2 - Stable Diffusion 1.5 generated Cézanne landscape paintings with guidance scale 8 (w/o textual inversion) - 88 images\
3 - Stable Diffusion 1.5 generated Cézanne landscape paintings with guidance scale 0-1 (w/o textual inversion) - 93 images\
4 - Impressionist landscape paintings from WikiArt dataset - 94 images\
\
Textual Inversion was not implemented when creating the dataset. A separate test dataset with Textual Inversion generated Cézanne landscape paintings was created. 

TIDS1 - images created by Stable Diffusion 2 with textual inversion trained prompt: "painting in the style of <Cezanne>"

TIDS2 - images created by Stable Diffusion 2 with textual inversion trained prompt: "landscape painting in the style of <Cezanne>"

TIDS3 - images created by Stable Diffusion 2 with textual inversion trained prompt: "painting of the Provence in the style of <Cézanne>"

TIDS4 - images created by Stable Diffusion 2 with textual inversion trained prompt: "painting of Mont Saint Victoire in the style of <Cézanne>"

Before training the CNN, the images were all resized to (512, 512, 3). This was done to disabuse the CNN from learning image sizes. Furthermore, the images were all converted to grayscale. This was done to disabuse the CNN from learning color schemes, as color is difficult to grasp consistently for cameras and depends on lighting etc..  

